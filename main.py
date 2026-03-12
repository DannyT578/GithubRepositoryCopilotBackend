"""
GitHub Chat Backend — FastAPI entrypoint
"""

import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager

# ProactorEventLoop is required for subprocess support (e.g. gitingest's git calls).
# WindowsProactorEventLoopPolicy only exists on Windows, so this is a no-op elsewhere.
if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from limiter import limiter
from routers import auth, repos

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger.info("Backend starting up")
    yield
    logger.info("Backend shutting down")


app = FastAPI(
    title="GitHub Chat API",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ---------------------------------------------------------------------------
# Global unhandled-exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception | %s %s | %s: %s\n%s",
        request.method,
        request.url.path,
        type(exc).__name__,
        exc,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


# ---------------------------------------------------------------------------
# CORS — allow the Next.js dev server, configured FRONTEND_URL, and
# any Vercel preview deployment URLs (*.vercel.app)
# ---------------------------------------------------------------------------
_frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_url],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(repos.router, prefix="/repos", tags=["repos"])


@app.get("/health")
async def health():
    return {"status": "ok"}
