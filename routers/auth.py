"""
Auth router
-----------
Appwrite handles the GitHub OAuth flow on the frontend.  After sign-in the
frontend receives the GitHub access token from the Appwrite session
(session.providerAccessToken) and sends it as:

    Authorization: Bearer <github_access_token>

This router:
  GET /auth/me  — verifies the bearer token against GitHub and returns the
                   user profile.

The get_access_token dependency is imported by repos.py to protect all
repository endpoints.
"""

import logging

import aiohttp
from fastapi import APIRouter, Depends, Header, HTTPException, Request

from limiter import limiter

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared dependency
# ---------------------------------------------------------------------------

def get_access_token(authorization: str | None = Header(default=None)) -> str:
    """Extracts and returns the raw GitHub access token from the Authorization
    header, or raises HTTP 401 if it is absent or malformed."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return authorization[7:].strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/me")
@limiter.limit("20/minute")
async def me(request: Request, token: str = Depends(get_access_token)):
    """Return the authenticated GitHub user's profile."""
    logger.info("GET /auth/me | verifying token")
    async with aiohttp.ClientSession() as session:
        resp = await session.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        if resp.status == 401:
            logger.warning("GET /auth/me | invalid or expired token")
            raise HTTPException(status_code=401, detail="Invalid or expired GitHub token.")
        if not resp.ok:
            logger.error("GET /auth/me | GitHub API error | status=%d", resp.status)
            raise HTTPException(status_code=resp.status, detail="GitHub API error.")
        user = await resp.json()

    logger.info("GET /auth/me | ok | login=%s", user.get("login"))
    return {
        "login": user["login"],
        "name": user.get("name"),
        "avatar_url": user.get("avatar_url"),
        "html_url": user.get("html_url"),
    }
