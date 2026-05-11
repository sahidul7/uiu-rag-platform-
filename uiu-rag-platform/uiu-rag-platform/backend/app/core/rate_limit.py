"""
Rate Limiting — prevents abuse by students
20 queries/minute per user, 100/minute per IP
"""
from __future__ import annotations
import time
from collections import defaultdict
from fastapi import HTTPException, Request, status

# In-memory store (use Redis in production for multi-worker)
_request_counts: dict[str, list[float]] = defaultdict(list)

def check_rate_limit(key: str, max_requests: int = 20, window_seconds: int = 60) -> None:
    now = time.time()
    window_start = now - window_seconds
    _request_counts[key] = [t for t in _request_counts[key] if t > window_start]
    if len(_request_counts[key]) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {max_requests} requests per minute.",
            headers={"Retry-After": "60"},
        )
    _request_counts[key].append(now)
