# GitHub Chat — Backend Documentation

FastAPI backend that ingests GitHub repositories into a FAISS vector index and answers
natural language questions about them using the user's chosen LLM (OpenAI, Google Gemini,
or Anthropic Claude).

> **Separate repos:** This document covers the **backend** only.
> See `DOCUMENTATION.md` in the frontend repo for the Next.js app.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Architecture Overview](#2-architecture-overview)
3. [Entry Point — `main.py`](#3-entry-point--mainpy)
4. [Rate Limiting — `limiter.py`](#4-rate-limiting--limiterpy)
5. [Auth Router — `routers/auth.py`](#5-auth-router--routersauthpy)
6. [Repository Router — `routers/repos.py`](#6-repository-router--routersrepospy)
7. [Ingestion Module — `modules/ingest.py`](#7-ingestion-module--modulesingestpy)
8. [Cache Module — `modules/cache.py`](#8-cache-module--modulescachepy)
9. [Vector Index + LLM Module — `modules/index.py`](#9-vector-index--llm-module--modulesindexpy)
10. [Configuration & Environment Variables](#10-configuration--environment-variables)
11. [Running Locally](#11-running-locally)
12. [Deploying to Railway](#12-deploying-to-railway)
13. [Data Flow Walkthrough](#13-data-flow-walkthrough)

---

## 1. Project Structure

```
backend/
├── main.py               ← FastAPI app, CORS, middleware, routers
├── limiter.py            ← slowapi rate limiter (Redis or in-memory)
├── run.py                ← dev entrypoint (uvicorn + Windows ProactorEventLoop fix)
├── requirements.txt      ← Python dependencies
├── railway.toml          ← Railway deployment config
├── .env                  ← local environment variables (never commit)
├── .env.example          ← template
├── routers/
│   ├── __init__.py
│   ├── auth.py           ← GET /auth/me
│   └── repos.py          ← POST /repos/ingest, GET /repos/{owner}/{repo},
│                            POST /repos/{owner}/{repo}/chat
└── modules/
    ├── __init__.py
    ├── ingest.py         ← gitingest wrapper
    ├── cache.py          ← LRU disk cache
    └── index.py          ← FAISS vector index + LiteLLM multi-provider LLM
```

---

## 2. Architecture Overview

```
Frontend (Next.js — Vercel)
    │
    │  Authorization: Bearer <github_access_token>  (every request)
    │  chat_api_key / embed_api_key in request body  (ingest + chat)
    ▼
FastAPI  :8000  (Railway)
    ├── slowapi rate limiting  (per-user, keyed by hashed Bearer token)
    ├── CORS  (FRONTEND_URL + *.vercel.app preview URLs)
    ├── GET  /auth/me         ← verify GitHub token, return user profile
    ├── POST /repos/ingest    ← ingest repo, build FAISS index
    ├── GET  /repos/{owner}/{repo}  ← live GitHub metadata
    └── POST /repos/{owner}/{repo}/chat  ← RAG chat answer
            │
            ├── gitingest   → raw repo text
            ├── LiteLLM     → embeddings  (OpenAI / Gemini / OpenAI-for-Anthropic)
            ├── FAISS       → in-process vector index
            ├── LiteLLM     → chat completion  (OpenAI / Gemini / Anthropic)
            └── disk cache  → JSON files in tempdir (6-hour TTL)
```

**Key design decisions:**
- **No server-side OAuth.** The frontend handles GitHub OAuth via Appwrite. The backend only receives the resulting GitHub access token as a Bearer header and verifies it by calling `api.github.com/user`.
- **User-supplied LLM keys.** API keys are sent in the request body and never stored server-side. This means the backend has no key management burden and users control their own quota.
- **In-process FAISS index.** The vector index lives in a Python dict keyed by `owner/repo:embed_model`. It survives across requests but resets on server restart — the disk cache allows transparent rebuilding without re-fetching from GitHub.
- **Multi-provider via LiteLLM.** A single `litellm.aembedding()` / `litellm.acompletion()` call routes to any supported provider. The backend never imports `openai` directly for chat.

---

## 3. Entry Point — `main.py`

Creates the FastAPI app and wires up all middleware and routers.

| Item | Detail |
|---|---|
| CORS origins | `FRONTEND_URL` env var + all `*.vercel.app` subdomains (via `allow_origin_regex`) |
| Rate limiting | `SlowAPIMiddleware` + `RateLimitExceeded` handler (returns HTTP 429) |
| Lifespan | Configures `logging.basicConfig` on startup |
| Global exception handler | Catches any unhandled exception, logs the full traceback, returns `500 {"detail": "..."}` |
| Health check | `GET /health` → `{"status": "ok"}` |

### Windows ProactorEventLoop fix (`run.py`)

`gitingest` uses `git` subprocesses internally. On Windows, uvicorn's reload worker spawns with `SelectorEventLoop` which cannot start subprocesses. `run.py` passes `loop="loop_factory:proactor_loop_factory"` to uvicorn so both the main and reloader processes use `ProactorEventLoop`.

This is only needed for local development with `--reload`. Railway runs without `--reload` so `uvicorn main:app` directly is used there (`railway.toml`).

---

## 4. Rate Limiting — `limiter.py`

Provides a shared `slowapi.Limiter` instance used by all routers.

### Key function

Rate limit buckets are **per authenticated user**, not per IP:
- If an `Authorization: Bearer <token>` header is present → key is `tok:<sha256 of token[:16]>`
- Otherwise falls back to client IP

This prevents users on shared networks (office, university) from hitting each other's limits.

### Storage backend

| `REDIS_URL` set? | Storage |
|---|---|
| Yes (`rediss://...`) | Redis — shared across all processes/workers, survives restarts |
| No | In-memory — per process only; fine for single-process local dev |

For production on Railway with a single dyno, either works. For multiple replicas, Redis is required.

### Limits applied

| Endpoint | Limit | Reason |
|---|---|---|
| `GET /auth/me` | 20/minute | Lightweight but called on every page load |
| `POST /repos/ingest` | 5/minute | Expensive: git fetch + embedding hundreds of chunks |
| `GET /repos/{owner}/{repo}` | 30/minute | GitHub API call per request |
| `POST /repos/{owner}/{repo}/chat` | 30/minute | LLM call per request |

---

## 5. Auth Router — `routers/auth.py`

**Purpose:** Verify the GitHub access token sent by the frontend and return the user's profile.

The backend does not implement GitHub OAuth itself — that is handled by Appwrite on the frontend. The backend only validates the resulting token.

### Shared dependency: `get_access_token`

```python
def get_access_token(authorization: str | None = Header(default=None)) -> str
```

Used by every protected endpoint in `repos.py`. Extracts the token from `Authorization: Bearer <token>`, raises HTTP 401 if missing or malformed.

### Endpoint

**`GET /auth/me`** (rate limited: 20/minute)

- Calls `GET https://api.github.com/user` with the token
- Returns `{login, name, avatar_url, html_url}`
- Returns HTTP 401 if the token is invalid or expired
- Returns HTTP 502 on GitHub API errors

---

## 6. Repository Router — `routers/repos.py`

All endpoints require `Authorization: Bearer <github_token>`.

### Endpoint: `POST /repos/ingest` (rate limited: 5/minute)

**Request body:**
```json
{
  "repo_url": "https://github.com/owner/repo",
  "chat_model": "gpt-4o-mini",
  "chat_api_key": "sk-...",
  "embed_api_key": ""
}
```

**Flow:**
1. Validate `repo_url` with regex `^https?://(www\.)?github\.com/[\w\-\.]+/[\w\-\.]+$`
2. `_check_repo_accessible()` — GET `api.github.com/repos/{owner}/{repo}` with ***user's*** token (works for private repos they have access to)
3. `_get_embed_config()` — select embedding model + key based on `chat_model`
4. `_ensure_index()` — check in-memory store → check disk cache → ingest + build if needed (asyncio-locked per repo)
5. Return GitHub metadata + chunk count

**Error codes returned as `detail`:**
| Code | HTTP | Meaning |
|---|---|---|
| `error:repo_not_found` | 404 | Repo doesn't exist or user has no access |
| `error:repo_too_large` | 413 | Exceeds 750K token limit |
| `error:repo_private` | 403 | Private repo with rate-limited unauthenticated access |

### Endpoint: `GET /repos/{owner}/{repo}` (rate limited: 30/minute)

Returns live GitHub metadata (stars, forks, language, updated_at) plus `indexed` / `cached` boolean flags indicating whether the repo is currently in memory and on disk.

### Endpoint: `POST /repos/{owner}/{repo}/chat` (rate limited: 30/minute)

**Request body:**
```json
{
  "question": "What does the auth module do?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "chat_model": "gpt-4o-mini",
  "chat_api_key": "sk-...",
  "embed_api_key": ""
}
```

**Flow:**
1. Look up `_index_store[owner/repo:embed_model]` — returns 400 if not ingested
2. Embed the question, FAISS search → top-k chunks
3. Build grounded prompt (summary + tree + chunks + history)
4. Call LLM via LiteLLM
5. Return `{answer, sources: [{type, file, lineStart, lineEnd, snippet}]}`

### Provider → embedding model mapping (`_get_embed_config`)

| Chat model prefix | Embedding model | Embedding key |
|---|---|---|
| `gpt-*` / OpenAI | `text-embedding-3-small` | `chat_api_key` |
| `gemini/` | `gemini/gemini-embedding-001` | `chat_api_key` |
| `claude-*` / Anthropic | `text-embedding-3-small` | `embed_api_key` (separate OpenAI key required) |

### In-memory index store

Two module-level dicts:
- `_index_store: dict[str, RepoIndex]` — keyed `"owner/repo:embed_model"`
- `_index_locks: dict[str, asyncio.Lock]` — per-repo lock preventing concurrent duplicate ingestions

---

## 7. Ingestion Module — `modules/ingest.py`

Wraps `gitingest` to convert a GitHub repository URL into three strings: `summary`, `tree`, and `content`.

### Functions

**`ingest_repo(repo_url)`** *(async)*
- Calls `gitingest.ingest_async()` with `exclude_patterns={"tests/*", "docs/*"}`
- Parses the token count from the gitingest summary string
- Raises `ValueError("error:repo_too_large")` if tokens exceed `MAX_REPO_SIZE_IN_K_TOKENS` (750K)
- Maps other exceptions to `error:repo_not_found` or `error:repo_private`
- Returns `(summary, tree, content)`

---

## 8. Cache Module — `modules/cache.py`

LRU disk cache for ingested repository content. Avoids re-fetching from GitHub on every server restart.

### Constants

| Constant | Value | Meaning |
|---|---|---|
| `CACHE_DIR` | `tempfile.gettempdir()/repo_cache` | Cross-platform temp dir |
| `CACHE_TTL_SECONDS` | 21 600 (6 hours) | Entry validity window |
| `CACHE_MAX_FILES` | 100 | Max cached repos before LRU eviction |

### Functions

**`load_repo_cache(owner, repo)`** → `dict | None`
Reads JSON, updates access time for LRU tracking, checks TTL. Returns `None` if missing or stale.

**`save_repo_cache(owner, repo, summary, tree, content)`**
Writes `{summary, tree, content, cached_at}` to JSON, then evicts excess files by `atime` (oldest first).

---

## 9. Vector Index + LLM Module — `modules/index.py`

Builds a semantic search index and generates grounded LLM answers. Supports all providers via LiteLLM.

### Data classes
- **`Chunk`** — `text`, `source` (filename), `start_line`, `end_line`
- **`RepoIndex`** — `chunks: list[Chunk]`, `index: faiss.IndexFlatIP`, `embed_model: str`

### Embedding pipeline

```
raw content string
       │
  _chunk_content()    ← splits on gitingest "File: …" headers,
       │                slides a window of CHUNK_SIZE lines with CHUNK_OVERLAP overlap
       ▼
list[Chunk]
       │
  _embed_texts()      ← batches texts (256 for OpenAI, 20 for Gemini),
       │                calls litellm.aembedding(), L2-normalises vectors
       ▼
numpy float32 matrix
       │
  faiss.IndexFlatIP   ← brute-force inner-product (= cosine after L2-norm)
       │
  RepoIndex
```

### Rate limit retry (`_with_backoff`)

Both `_embed_texts()` and the LLM completion call are wrapped in `_with_backoff()`:
- Catches `litellm.exceptions.RateLimitError` (HTTP 429 from provider)
- Retries up to **5 times** with exponential backoff: 2s → 4s → 8s → 16s → 32s
- On all retries exhausted, re-raises the original error

### Chat flow (`chat_with_repo`)

1. Embed the question with the same model used at index time
2. FAISS search → top-k `Chunk` objects
3. Build system prompt: repo summary + directory tree (truncated to 3 000 chars) + retrieved chunks with file/line citations
4. Append last 10 turns of conversation history
5. `litellm.acompletion()` with `temperature=0.2`, `max_tokens=1024`
6. Return `(answer_text, source_chunks)`

### Tunable env vars

| Var | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | `120` | Lines per chunk |
| `CHUNK_OVERLAP` | `20` | Overlap between consecutive chunks |
| `TOP_K` | `6` | Chunks retrieved per query |

---

## 10. Configuration & Environment Variables

### `backend/.env` (copy from `.env.example`)

| Variable | Required | Description |
|---|---|---|
| `FRONTEND_URL` | ✅ | URL of the Next.js frontend. Used for CORS. Local: `http://localhost:3000`. Production: your Vercel URL |
| `REDIS_URL` | optional | Redis connection URL for rate limiting. Falls back to in-memory if unset. Upstash format: `rediss://default:<token>@<host>.upstash.io:6379` |
| `CHUNK_SIZE` | optional | Lines per chunk (default `120`) |
| `CHUNK_OVERLAP` | optional | Chunk overlap (default `20`) |
| `TOP_K` | optional | Retrieved chunks per query (default `6`) |

> **No API keys stored here.** LLM API keys are sent by the user in each request body and never persisted server-side.

---

## 11. Running Locally

### Requirements
- Python 3.11+
- Git (must be on `PATH` — required by gitingest)

```powershell
cd backend

# First time only
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env: set FRONTEND_URL=http://localhost:3000

# Every time
.\.venv\Scripts\Activate.ps1
python run.py
```

Verify:
- Health: `http://localhost:8000/health` → `{"status":"ok"}`
- Interactive API docs: `http://localhost:8000/docs`

---

## 12. Deploying to Railway

`railway.toml` at the backend root configures the deployment automatically.

**Steps:**

1. [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
2. Set **Root Directory** to `backend` in the Railway service settings
3. Railway detects Python via `requirements.txt` and uses `railway.toml` for the start command
4. Add environment variables in Railway's dashboard:

| Key | Value |
|---|---|
| `FRONTEND_URL` | `https://your-app.vercel.app` |
| `REDIS_URL` | `rediss://default:<token>@<host>.upstash.io:6379` |

5. Copy the Railway public URL (e.g. `https://your-app.up.railway.app`)
6. Set `NEXT_PUBLIC_BACKEND_URL` in Vercel to this URL

**Why not Vercel for the backend:**
Vercel is serverless — each request runs in a fresh process. The in-memory FAISS `_index_store` would be wiped on every invocation, making every chat request fail with "not ingested yet". Railway runs a persistent process, which is required for in-memory state to survive between requests.

---

## 13. Data Flow Walkthrough

### Repository ingestion (`POST /repos/ingest`)

```
Client → POST /repos/ingest  { repo_url, chat_model, chat_api_key, embed_api_key }
              │
              ├── validate repo_url (regex)
              ├── _check_repo_accessible(repo_url, token)
              │     └── GET api.github.com/repos/{owner}/{repo}  (user's token)
              ├── _get_embed_config(chat_model, ...)  → embed_model, embed_key
              └── _ensure_index(store_key, owner, repo, token, embed_model, embed_key)
                    ├── check _index_store  (memory hit → return immediately)
                    ├── check disk cache    (cache hit → rebuild index from cached text)
                    └── ingest from GitHub:
                          ├── ingest_repo(url)         ← gitingest (git clone/fetch)
                          ├── save_repo_cache(...)     ← write JSON to disk
                          └── build_index(content, embed_model, embed_key)
                                ├── _chunk_content()  ← split into ~120-line chunks
                                └── _embed_texts()    ← litellm.aembedding() in batches
                                                         with exponential backoff on 429
```

### Chat (`POST /repos/{owner}/{repo}/chat`)

```
Client → POST /repos/{owner}/{repo}/chat  { question, history, chat_model, chat_api_key }
              │
              ├── look up _index_store[store_key]  (400 if not ingested)
              ├── load disk cache for summary + tree
              └── chat_with_repo(...)
                    ├── query_index(question)
                    │     ├── _embed_texts([question])   ← single embedding call
                    │     └── faiss.search(q_vec, TOP_K) ← top-k chunks
                    ├── build system prompt (summary + tree + chunks)
                    ├── append history[-10:]
                    └── litellm.acompletion()            ← with exponential backoff on 429
                          → {answer, sources}
```
