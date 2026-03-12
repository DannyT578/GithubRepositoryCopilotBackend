"""
Repository router
-----------------
POST /repos/ingest          — ingest a repo (uses the user's GitHub token)
GET  /repos/{owner}/{repo}  — return cached repo metadata
POST /repos/{owner}/{repo}/chat — stream a chat answer grounded in the index

All endpoints require a valid session cookie (the GitHub access token).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Annotated, AsyncGenerator

logger = logging.getLogger(__name__)

import aiohttp
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, field_validator

from limiter import limiter

from modules.cache import load_repo_cache, save_repo_cache
from modules.ingest import ingest_repo
from modules.index import RepoIndex, build_index, chat_with_repo
from routers.auth import get_access_token

router = APIRouter()

# ---------------------------------------------------------------------------
# In-process index store  {owner/repo: RepoIndex}
# ---------------------------------------------------------------------------
_index_store: dict[str, RepoIndex] = {}
_index_locks: dict[str, asyncio.Lock] = {}


def _get_lock(key: str) -> asyncio.Lock:
    if key not in _index_locks:
        _index_locks[key] = asyncio.Lock()
    return _index_locks[key]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    repo_url: str
    chat_model: str
    chat_api_key: str
    embed_api_key: str = ""  # only needed when chat provider has no embeddings (e.g. Anthropic)

    @field_validator("repo_url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        pattern = r"^https?://(www\.)?github\.com/[\w\-\.]+/[\w\-\.]+$"
        if not re.match(pattern, v):
            raise ValueError("Must be a valid GitHub repository URL.")
        return v


class ChatRequest(BaseModel):
    question: str
    history: list[dict] | None = None
    chat_model: str
    chat_api_key: str
    embed_api_key: str = ""  # only needed when chat provider has no embeddings (e.g. Anthropic)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_owner_repo(repo_url: str) -> tuple[str, str]:
    parts = repo_url.rstrip("/").split("/")
    return parts[-2], parts[-1]


def _get_embed_config(chat_model: str, chat_api_key: str, embed_api_key: str) -> tuple[str, str]:
    """
    Return (embed_model, embed_key) based on the chat model provider.
    - OpenAI models → OpenAI text-embedding-3-small, same key
    - Gemini models → Google gemini-embedding-001, same key
    - Anthropic/Claude → OpenAI embeddings with embed_api_key (separate key required)
    """
    if "gemini" in chat_model.lower():
        return "gemini/gemini-embedding-001", chat_api_key
    elif chat_model.startswith("claude"):
        return "text-embedding-3-small", embed_api_key or chat_api_key
    else:
        # Default: OpenAI family (gpt-*, etc.)
        return "text-embedding-3-small", chat_api_key


async def _check_repo_accessible(repo_url: str, token: str) -> dict:
    """Use the caller's token to verify the repo exists and is accessible."""
    owner, repo = _parse_owner_repo(repo_url)
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    logger.info("_check_repo_accessible | GET %s", api_url)
    async with aiohttp.ClientSession() as session:
        resp = await session.get(
            api_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        if resp.status == 404:
            logger.warning("_check_repo_accessible | 404 | %s", api_url)
            raise HTTPException(status_code=404, detail="Repository not found or you don't have access.")
        if resp.status == 403:
            logger.warning("_check_repo_accessible | 403 | %s", api_url)
            raise HTTPException(status_code=403, detail="Access denied. Check your GitHub permissions.")
        if resp.status != 200:
            logger.error("_check_repo_accessible | unexpected status=%d | %s", resp.status, api_url)
            raise HTTPException(status_code=502, detail="GitHub API error.")
        logger.info("_check_repo_accessible | ok | %s", api_url)
        return await resp.json()


async def _ensure_index(
    store_key: str, owner: str, repo: str, token: str,
    embed_model: str, embed_api_key: str,
) -> RepoIndex:
    """Return a cached index or build a new one (thread-safe)."""
    if store_key in _index_store:
        logger.info("_ensure_index | memory-cache hit | %s", store_key)
        return _index_store[store_key]

    async with _get_lock(store_key):
        if store_key in _index_store:
            logger.info("_ensure_index | memory-cache hit (post-lock) | %s", store_key)
            return _index_store[store_key]

        owner_repo = f"{owner}/{repo}"
        cached = load_repo_cache(owner, repo)
        if cached:
            logger.info("_ensure_index | disk-cache hit | %s", owner_repo)
            summary = cached["summary"]
            tree = cached["tree"]
            content = cached["content"]
        else:
            logger.info("_ensure_index | ingesting from GitHub | %s", owner_repo)
            summary, tree, content = await ingest_repo(
                f"https://github.com/{owner}/{repo}"
            )
            save_repo_cache(owner, repo, summary, tree, content)

        logger.info("_ensure_index | building vector index | %s", store_key)
        idx = await build_index(content, embed_model=embed_model, embed_api_key=embed_api_key)
        _index_store[store_key] = idx
        logger.info("_ensure_index | index ready | %s | chunks=%d", store_key, len(idx.chunks))
        return idx


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/ingest")
@limiter.limit("5/minute")
async def ingest(
    request: Request,
    body: IngestRequest,
    token: Annotated[str, Depends(get_access_token)],
):
    """Ingest a GitHub repository using the user's token."""
    gh_meta = await _check_repo_accessible(body.repo_url, token)

    owner, repo = _parse_owner_repo(body.repo_url)
    embed_model, embed_key = _get_embed_config(body.chat_model, body.chat_api_key, body.embed_api_key)
    store_key = f"{owner}/{repo}:{embed_model}"
    logger.info("POST /repos/ingest | %s | model=%s embed=%s", f"{owner}/{repo}", body.chat_model, embed_model)

    # Run ingestion + indexing (possibly cached)
    try:
        idx = await _ensure_index(store_key, owner, repo, token, embed_model, embed_key)
    except ValueError as exc:
        code = str(exc)
        logger.warning("POST /repos/ingest | ingest error | %s | code=%s", key, code)
        status_map = {
            "error:repo_not_found": 404,
            "error:repo_too_large": 413,
            "error:repo_private":   403,
        }
        raise HTTPException(
            status_code=status_map.get(code, 400),
            detail=code,
        )
    except Exception as exc:
        logger.error("POST /repos/ingest | unexpected error | %s | %s: %s", f"{owner}/{repo}", type(exc).__name__, exc)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    return {
        "owner": owner,
        "repo": repo,
        "full_name": gh_meta.get("full_name"),
        "description": gh_meta.get("description"),
        "stars": gh_meta.get("stargazers_count"),
        "forks": gh_meta.get("forks_count"),
        "language": gh_meta.get("language"),
        "updated_at": gh_meta.get("updated_at"),
        "html_url": gh_meta.get("html_url"),
        "chunks": len(idx.chunks),
    }


@router.get("/{owner}/{repo}")
@limiter.limit("30/minute")
async def get_repo_info(
    request: Request,
    owner: str,
    repo: str,
    token: Annotated[str, Depends(get_access_token)],
):
    """Return GitHub metadata for an already-ingested repository."""
    gh_meta = await _check_repo_accessible(
        f"https://github.com/{owner}/{repo}", token
    )
    key = f"{owner}/{repo}"
    indexed = key in _index_store
    cached = load_repo_cache(owner, repo) is not None

    return {
        "owner": owner,
        "repo": repo,
        "full_name": gh_meta.get("full_name"),
        "description": gh_meta.get("description"),
        "stars": gh_meta.get("stargazers_count"),
        "forks": gh_meta.get("forks_count"),
        "language": gh_meta.get("language"),
        "updated_at": gh_meta.get("updated_at"),
        "html_url": gh_meta.get("html_url"),
        "indexed": indexed,
        "cached": cached,
    }


@router.post("/{owner}/{repo}/chat")
@limiter.limit("30/minute")
async def chat(
    request: Request,
    owner: str,
    repo: str,
    body: ChatRequest,
    token: Annotated[str, Depends(get_access_token)],
):
    """Answer a question about an ingested repository using the vector index."""
    embed_model, embed_key = _get_embed_config(body.chat_model, body.chat_api_key, body.embed_api_key)
    store_key = f"{owner}/{repo}:{embed_model}"
    if store_key not in _index_store:
        logger.warning("POST /repos/%s/%s/chat | not ingested yet | embed=%s", owner, repo, embed_model)
        raise HTTPException(
            status_code=400,
            detail="Repository has not been ingested yet. Call POST /repos/ingest first.",
        )

    logger.info("POST /repos/%s/%s/chat | question=%r | model=%s", owner, repo, body.question[:80], body.chat_model)
    cached = load_repo_cache(owner, repo)
    summary = cached["summary"] if cached else ""
    tree = cached["tree"] if cached else ""

    try:
        answer, source_chunks = await chat_with_repo(
            repo_index=_index_store[store_key],
            question=body.question,
            chat_model=body.chat_model,
            chat_api_key=body.chat_api_key,
            embed_model=embed_model,
            embed_api_key=embed_key,
            history=body.history,
            tree=tree,
            summary=summary,
        )
    except Exception as exc:
        logger.error("POST /repos/%s/%s/chat | chat error | %s: %s", owner, repo, type(exc).__name__, exc)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    sources = [
        {
            "type": "code",
            "file": chunk.source,
            "lineStart": chunk.start_line,
            "lineEnd": chunk.end_line,
            "snippet": chunk.text[:200],
        }
        for chunk in source_chunks
    ]

    return {"answer": answer, "sources": sources}
