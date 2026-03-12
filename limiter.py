"""
Rate limiter
------------
Shared slowapi Limiter instance used by all routers.
Rate limit keys are based on the GitHub token (hashed) so each authenticated
user gets their own bucket.  Unauthenticated requests fall back to IP.

Storage:
  - If REDIS_URL is set (e.g. rediss://... for Upstash) → Redis backend
  - Otherwise → in-memory (fine for single-process local dev)
"""

import hashlib
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _rate_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        # Hash token so it never appears in logs
        return "tok:" + hashlib.sha256(auth[7:].encode()).hexdigest()[:16]
    return get_remote_address(request)


_redis_url = os.getenv("REDIS_URL", "memory://")

limiter = Limiter(key_func=_rate_key, storage_uri=_redis_url)
