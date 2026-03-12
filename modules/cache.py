"""
Repository Cache Module

Provides LRU caching for ingested GitHub repositories to improve performance
and reduce redundant API calls.
"""

import logging
import os
import json
import tempfile
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Cache directory — cross-platform temp folder
CACHE_DIR = os.path.join(tempfile.gettempdir(), "repo_cache")

# Cache expiration time (6 hours)
CACHE_TTL_SECONDS = 6 * 60 * 60

# Maximum number of cached repositories before LRU eviction
CACHE_MAX_FILES = 100


def _enforce_lru_cache_limit() -> None:
    """
    Enforce LRU (Least Recently Used) cache size limit.
    
    Removes oldest accessed cache files when limit is exceeded.
    Uses file access time to determine which files to evict.
    """
    files = [
        (f, os.path.getatime(os.path.join(CACHE_DIR, f)))
        for f in os.listdir(CACHE_DIR)
        if f.endswith(".json")
    ]
    
    if len(files) > CACHE_MAX_FILES:
        # Sort by access time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Remove oldest files to get back to limit
        files_to_remove = files[:len(files) - CACHE_MAX_FILES]
        for filename, _ in files_to_remove:
            os.remove(os.path.join(CACHE_DIR, filename))


def get_cache_path(owner: str, repo: str) -> str:
    """
    Get the file path for a repository's cache file.
    
    Args:
        owner: Repository owner username
        repo: Repository name
    
    Returns:
        Full path to the cache file
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{owner}_{repo}.json")


def load_repo_cache(owner: str, repo: str) -> Optional[Dict[str, Any]]:
    """
    Load cached repository data if it exists and is not expired.
    
    Args:
        owner: Repository owner username
        repo: Repository name
    
    Returns:
        Cached data dict containing summary, tree, content, and cached_at timestamp.
        Returns None if cache doesn't exist or is expired.
    """
    path = get_cache_path(owner, repo)
    
    if not os.path.exists(path):
        logger.debug("load_repo_cache | miss | %s/%s", owner, repo)
        return None

    try:
        os.utime(path, None)
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("load_repo_cache | read error | %s/%s | %s: %s", owner, repo, type(exc).__name__, exc)
        return None

    cached_at = data.get("cached_at", 0)
    age_h = (time.time() - cached_at) / 3600
    if time.time() - cached_at < CACHE_TTL_SECONDS:
        logger.info("load_repo_cache | hit | %s/%s | age=%.1fh", owner, repo, age_h)
        return data

    logger.info("load_repo_cache | expired | %s/%s | age=%.1fh", owner, repo, age_h)
    return None


def save_repo_cache(owner: str, repo: str, summary: Any, tree: Any, content: Any) -> None:
    """
    Save repository data to cache.
    
    Args:
        owner: Repository owner username
        repo: Repository name
        summary: Repository summary metadata
        tree: Repository directory structure
        content: Repository file contents
    """
    path = get_cache_path(owner, repo)
    
    cache_data = {
        "summary": summary,
        "tree": tree,
        "content": content,
        "cached_at": time.time()
    }
    
    try:
        with open(path, "w") as f:
            json.dump(cache_data, f)
        logger.info("save_repo_cache | saved | %s/%s", owner, repo)
    except Exception as exc:
        logger.error("save_repo_cache | write error | %s/%s | %s: %s", owner, repo, type(exc).__name__, exc)
        raise

    _enforce_lru_cache_limit()


if __name__ == "__main__":
    """Test the cache module by running: python cache.py"""
    
    # Test cache operations
    test_owner = "testuser"
    test_repo = "testrepo"
    
    print("Testing cache operations...")
    print("-" * 60)
    
    # Test save
    print(f"\n1. Saving cache for {test_owner}/{test_repo}")
    save_repo_cache(
        test_owner,
        test_repo,
        summary="Test summary",
        tree="Test tree structure",
        content="Test content"
    )
    print("✓ Cache saved")
    
    # Test load
    print(f"\n2. Loading cache for {test_owner}/{test_repo}")
    cached_data = load_repo_cache(test_owner, test_repo)
    if cached_data:
        print("✓ Cache loaded successfully")
        print(f"   Summary: {cached_data['summary']}")
        print(f"   Tree: {cached_data['tree']}")
        print(f"   Content: {cached_data['content']}")
        print(f"   Cached at: {cached_data['cached_at']}")
    else:
        print("✗ Cache not found or expired")
    
    # Test non-existent cache
    print(f"\n3. Loading non-existent cache")
    cached_data = load_repo_cache("nonexistent", "repo")
    if cached_data:
        print("✗ Unexpected cache found")
    else:
        print("✓ Correctly returned None for non-existent cache")
    
    print("\n" + "-" * 60)
    print("Cache tests complete!")

