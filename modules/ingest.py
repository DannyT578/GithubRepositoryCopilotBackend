"""
GitHub Repository Ingestion Module

Converts GitHub repositories into LLM-friendly text format with validation
and token count limits.
"""

import logging

from gitingest import ingest_async  # type: ignore
import aiohttp
from typing import Tuple

logger = logging.getLogger(__name__)

# Maximum repository size in thousands of tokens
MAX_REPO_SIZE_IN_K_TOKENS = 750.0

# Patterns to exclude from ingestion
DEFAULT_EXCLUDE_PATTERNS = {"tests/*", "docs/*"}


async def check_repo_exists(repo_url: str) -> bool:
    """
    Check if a GitHub repository exists and is publicly accessible.
    """
    api_url = repo_url.replace("github.com", "api.github.com/repos")
    logger.debug("check_repo_exists | GET %s", api_url)

    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(api_url)
            exists = response.status == 200
            logger.debug("check_repo_exists | status=%d exists=%s", response.status, exists)
            return exists
        except Exception as exc:
            logger.warning("check_repo_exists | request failed for %s: %s", repo_url, exc)
            return False


def _parse_token_count(summary: str) -> float:
    """
    Extract token count from summary string.
    
    Args:
        summary: Summary string containing "Estimated tokens: 123K" or "1.2M"
    
    Returns:
        Token count in thousands (K). Returns 0.0 if not found.
    """
    if "Estimated tokens: " not in summary:
        return 0.0
    
    tokens_str = summary.split("Estimated tokens: ")[-1].strip()
    
    if tokens_str.endswith("M"):
        return float(tokens_str[:-1]) * 1000.0
    elif tokens_str.endswith("K"):
        return float(tokens_str[:-1])
    
    return 0.0


def _check_token_limit(summary: str) -> None:
    """
    Validate repository size is within token limits.
    
    Raises:
        ValueError: If repository exceeds 750K token limit
    """
    token_count_in_k = _parse_token_count(summary)
    
    if token_count_in_k > MAX_REPO_SIZE_IN_K_TOKENS:
        raise ValueError("error:repo_too_large")


async def ingest_repo(repo_url: str) -> Tuple[str, str, str]:
    """
    Convert a GitHub repository into LLM-friendly text format.

    Raises:
        ValueError with error codes:
            - "error:repo_not_found": Repository doesn't exist
            - "error:repo_too_large": Repository exceeds 750K token limit
            - "error:repo_private": Repository is private or rate limit exceeded
    """
    logger.info("ingest_repo | start | url=%s", repo_url)

    # Validate repository exists
    if not await check_repo_exists(repo_url):
        logger.warning("ingest_repo | repo not found | url=%s", repo_url)
        raise ValueError("error:repo_not_found")

    try:
        logger.info("ingest_repo | calling gitingest | url=%s", repo_url)
        summary, tree, content = await ingest_async(
            repo_url,
            exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
        )
        logger.info(
            "ingest_repo | gitingest complete | url=%s | content_len=%d",
            repo_url,
            len(content),
        )

        # Check size limits
        token_k = _parse_token_count(summary)
        logger.info("ingest_repo | token_count=%.1fK limit=%.1fK | url=%s", token_k, MAX_REPO_SIZE_IN_K_TOKENS, repo_url)
        _check_token_limit(summary)

        logger.info("ingest_repo | success | url=%s", repo_url)
        return summary, tree, content

    except ValueError:
        raise

    except Exception as exc:
        error_message = str(exc)
        logger.error(
            "ingest_repo | unexpected error | url=%s | %s: %s",
            repo_url,
            type(exc).__name__,
            exc,
        )

        if "Repository not found" in error_message or "Not Found" in error_message:
            raise ValueError("error:repo_not_found")

        if "Bad credentials" in error_message or "API rate limit exceeded" in error_message:
            raise ValueError("error:repo_private")

        raise


if __name__ == "__main__":
    """Test the ingestion module by running: python ingest.py"""
    import asyncio
    
    async def test_ingestion():
        """Test repository ingestion."""
        test_repo_url = "https://github.com/username/repository"
        
        print(f"Testing ingestion of: {test_repo_url}")
        print("-" * 60)
        
        try:
            summary, tree, content = await ingest_repo(test_repo_url)
            
            print("✓ Ingestion successful!")
            print("\nSummary:")
            print(summary)
            print("\n" + "=" * 60)
            print("\nDirectory Tree:")
            print(tree[:500] + "..." if len(tree) > 500 else tree)
            print("\n" + "=" * 60)
            print(f"\nContent length: {len(content)} characters")
            print(f"Content preview: {content[:200]}...")
            
        except ValueError as e:
            error_code = str(e)
            if error_code == "error:repo_not_found":
                print("✗ Error: Repository not found or not accessible")
            elif error_code == "error:repo_too_large":
                print("✗ Error: Repository is too large (exceeds 750K tokens)")
            elif error_code == "error:repo_private":
                print("✗ Error: Repository is private or API rate limit exceeded")
            else:
                print(f"✗ Error: {error_code}")
                
        except Exception as e:
            print(f"✗ Unexpected error: {type(e).__name__}: {e}")
    
    asyncio.run(test_ingestion())

