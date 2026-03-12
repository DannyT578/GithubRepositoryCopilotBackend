"""
Vector Index + LLM Query Module
---------------------------------
Builds an in-memory FAISS vector index from ingested repository content and
queries it with semantic search to inject *grounded* context into an LLM prompt.

Design
------
* The raw repository text is chunked by file boundaries (±overlap lines).
* Each chunk is embedded via LiteLLM (supports OpenAI, Google Gemini).
* At query time the top-k most similar chunks are retrieved and passed to
  the chosen LLM (OpenAI, Anthropic, Google Gemini) as grounded context.
* Index objects are stored keyed to owner/repo + embed_model so different
  embedding configurations coexist without conflict.
"""

from __future__ import annotations

import asyncio
import logging
import os
import textwrap
from dataclasses import dataclass
from typing import Optional

import numpy as np
import faiss                           # faiss-cpu
import litellm
from litellm.exceptions import RateLimitError

# Suppress litellm startup/debug noise
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "120"))   # lines per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
TOP_K         = int(os.getenv("TOP_K", "6"))

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    source: str          # e.g. "src/components/Button.tsx"
    start_line: int
    end_line: int


@dataclass
class RepoIndex:
    chunks: list[Chunk]
    index: faiss.IndexFlatIP   # inner-product (cosine after L2-norm)
    embed_model: str = "text-embedding-3-small"  # model used to build this index


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------
def _chunk_content(content: str) -> list[Chunk]:
    """
    Split flat ingested content (gitingest format) into per-file chunks.

    gitingest uses a header like:
        ================================================
        File: src/foo.ts
        ================================================
        <code>
    """
    chunks: list[Chunk] = []
    current_source = "unknown"
    current_lines: list[str] = []
    start_line = 1
    global_line = 0

    def flush(source: str, lines: list[str], s: int) -> None:
        if not lines:
            return
        for i in range(0, len(lines), CHUNK_SIZE - CHUNK_OVERLAP):
            block = lines[i : i + CHUNK_SIZE]
            if block:
                chunks.append(Chunk(
                    text="\n".join(block),
                    source=source,
                    start_line=s + i,
                    end_line=s + i + len(block) - 1,
                ))

    for raw_line in content.splitlines():
        global_line += 1
        stripped = raw_line.strip()

        if stripped.startswith("File: ") and len(stripped) < 300:
            # New file header encountered
            flush(current_source, current_lines, start_line)
            current_source = stripped[len("File: "):]
            current_lines = []
            start_line = global_line + 1
        elif stripped.startswith("=" * 20):
            # Separator line — skip
            pass
        else:
            current_lines.append(raw_line)

    flush(current_source, current_lines, start_line)
    return chunks


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
async def _with_backoff(coro_fn, *, max_retries: int = 5, base_delay: float = 2.0):
    """Call coro_fn() with exponential backoff on RateLimitError."""
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except RateLimitError as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16, 32 seconds
            logger.warning(
                "_with_backoff | RateLimitError attempt %d/%d | retrying in %.0fs | %s",
                attempt + 1, max_retries, delay, exc,
            )
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
async def _embed_texts(texts: list[str], embed_model: str, api_key: str) -> np.ndarray:
    """Return L2-normalised float32 embeddings, shape (n, dim)."""
    # Gemini has tight per-minute quotas — use smaller batches to stay under them
    BATCH = 20 if "gemini" in embed_model.lower() else 256
    all_vecs: list[list[float]] = []
    logger.info("_embed_texts | embedding %d texts with %s in batches of %d", len(texts), embed_model, BATCH)
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        try:
            resp = await _with_backoff(
                lambda b=batch: litellm.aembedding(model=embed_model, input=b, api_key=api_key)
            )
        except Exception as exc:
            logger.error(
                "_embed_texts | embedding error | model=%s batch_start=%d | %s: %s",
                embed_model, i, type(exc).__name__, exc,
            )
            raise
        all_vecs.extend(item.embedding for item in resp.data)

    mat = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return mat / norms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def build_index(content: str, embed_model: str, embed_api_key: str) -> RepoIndex:
    """Chunk + embed repo content and return a ready-to-query RepoIndex."""
    logger.info("build_index | start | content_len=%d embed_model=%s", len(content), embed_model)
    chunks = _chunk_content(content)
    if not chunks:
        logger.error("build_index | no indexable content found")
        raise ValueError("No indexable content found in repository.")
    logger.info("build_index | chunked into %d chunks", len(chunks))

    try:
        texts = [c.text for c in chunks]
        vecs = await _embed_texts(texts, embed_model, embed_api_key)
    except Exception as exc:
        logger.error("build_index | embedding failed | %s: %s", type(exc).__name__, exc)
        raise

    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    logger.info("build_index | FAISS index built | dim=%d vectors=%d", dim, len(chunks))

    return RepoIndex(chunks=chunks, index=idx, embed_model=embed_model)


async def query_index(
    repo_index: RepoIndex,
    question: str,
    embed_model: str,
    embed_api_key: str,
    top_k: int = TOP_K,
) -> list[Chunk]:
    """Retrieve top-k most relevant chunks for *question*."""
    logger.debug("query_index | question=%r top_k=%d", question[:80], top_k)
    try:
        q_vec = await _embed_texts([question], embed_model, embed_api_key)
        _scores, indices = repo_index.index.search(q_vec, top_k)
    except Exception as exc:
        logger.error("query_index | search failed | %s: %s", type(exc).__name__, exc)
        raise
    results = [repo_index.chunks[i] for i in indices[0] if i < len(repo_index.chunks)]
    logger.debug("query_index | returning %d chunks", len(results))
    return results


async def chat_with_repo(
    repo_index: RepoIndex,
    question: str,
    chat_model: str,
    chat_api_key: str,
    embed_model: str,
    embed_api_key: str,
    history: Optional[list[dict]] = None,
    tree: str = "",
    summary: str = "",
) -> tuple[str, list[Chunk]]:
    """
    Query the index, build a grounded prompt, call the chosen LLM, return
    (answer_text, source_chunks).  Supports any model LiteLLM can route to.
    """
    relevant_chunks = await query_index(repo_index, question, embed_model, embed_api_key)

    context_parts = []
    if summary:
        context_parts.append(f"Repository summary:\n{summary}")
    if tree:
        context_parts.append(f"Directory tree:\n{textwrap.shorten(tree, 3000, placeholder='...')}")
    for chunk in relevant_chunks:
        context_parts.append(
            f"--- {chunk.source} (lines {chunk.start_line}–{chunk.end_line}) ---\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are an expert software engineer helping a user understand a GitHub repository. "
        "Answer the user's question accurately and concisely using ONLY the provided context. "
        "If the answer is not in the context, say so. "
        "When referencing code, cite the file name.\n\n"
        f"CONTEXT:\n{context}"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-10:])          # keep last 10 turns for context
    messages.append({"role": "user", "content": question})

    logger.info("chat_with_repo | calling %s | question=%r", chat_model, question[:80])
    try:
        response = await _with_backoff(
            lambda: litellm.acompletion(
                model=chat_model,
                messages=messages,
                api_key=chat_api_key,
                temperature=0.2,
                max_tokens=1024,
            )
        )
    except Exception as exc:
        logger.error("chat_with_repo | LLM call failed | %s: %s", type(exc).__name__, exc)
        raise

    answer = response.choices[0].message.content or ""
    logger.info("chat_with_repo | done | answer_len=%d sources=%d", len(answer), len(relevant_chunks))
    return answer, relevant_chunks
