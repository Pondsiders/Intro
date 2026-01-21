"""
Cortex client for Intro.

Handles memory search with deduplication.
"""

import logging
import os

import httpx

from .models import MemoryItem

logger = logging.getLogger(__name__)

CORTEX_BASE_URL = os.environ.get("CORTEX_BASE_URL", "http://alpha-pi:7867")
CORTEX_API_KEY = os.environ.get("CORTEX_API_KEY", "")


async def search(query: str, limit: int = 3) -> list[MemoryItem]:
    """
    Search Cortex for memories matching a query.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        List of MemoryItem objects
    """
    if not CORTEX_API_KEY:
        logger.warning("CORTEX_API_KEY not set, skipping search")
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CORTEX_BASE_URL.rstrip('/')}/search",
                json={"query": query, "limit": limit},
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": CORTEX_API_KEY,
                },
            )
            response.raise_for_status()

        data = response.json()
        memories = []
        for item in data.get("memories", []):
            memories.append(MemoryItem(
                id=item["id"],
                content=item["content"],
                created_at=item.get("created_at", ""),
            ))

        return memories

    except Exception as e:
        logger.error(f"Cortex search failed: {e}")
        return []


async def search_with_dedup(
    queries: list[str],
    seen_ids: set[int],
    limit_per_query: int = 3,
) -> tuple[list[tuple[str, MemoryItem]], set[int]]:
    """
    Search Cortex for multiple queries, deduplicating results.

    Args:
        queries: List of search query strings
        seen_ids: Set of memory IDs already seen this session
        limit_per_query: Max results per query

    Returns:
        (list of (query, memory) tuples, updated seen_ids set)
    """
    results: list[tuple[str, MemoryItem]] = []
    new_seen = seen_ids.copy()

    for query in queries:
        memories = await search(query, limit=limit_per_query)
        for mem in memories:
            if mem.id not in new_seen:
                results.append((query, mem))
                new_seen.add(mem.id)
                break  # Take top 1 fresh result per query

    return results, new_seen
