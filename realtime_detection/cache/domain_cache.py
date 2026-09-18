from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """
    Stores one cached value and the time when it expires.
    """

    value: T
    expires_at: datetime


class DomainCache(Generic[T]):
    

    def __init__(self, ttl_seconds: int=86400) -> None:
        """
        Create a domain cache.

        ttl_seconds:
            Number of seconds for which a cached result remains valid.
            Default = 24 hours.
        """

        if ttl_seconds <=0:
            raise ValueError("ttl_seconds must be greater than 0")
        
        self.ttl = timedelta(seconds=ttl_seconds)

        self._cache: dict[str, CacheEntry[T]] = {}

    
    @staticmethod

    def normalize_domain(domain: str) -> str:
        """
        Normalize a domain before using it as a cache key.
        """

        return domain.strip().lower().rstrip(".")
    

    def get(self, domain: str) -> T | None:
        """
        Return a cached value if it exists and has not expired.

        Returns None when:
        - the domain is not cached
        - the cached entry has expired
        """

        key = self.normalize_domain(domain)

        entry = self._cache.get(key)

        if entry is None:
            return None 


        now = datetime.now(timezone.utc)


        if now >= entry.expires_at:
            del self._cache[key]
            return None


        return entry.value
     

    def set(self, domain: str, value: T) -> None:
        """
        Store a value in the cache.
        """

        key = self.normalize_domain(domain)

        expires_at = (
            datetime.now(timezone.utc)
            + self.ttl
        )

        self._cache[key] = CacheEntry(
            value=value,
            expires_at=expires_at,
        )

    
    def delete(self, domain: str) -> None:
        """
        Remove a domain from the cache if it exists.
        """

        key = self.normalize_domain(domain)

        self._cache.pop(key, None)


    def clear(self) -> None:
        """
        Return every cached entry
        """

        self._cache.clear()

    def size(self) -> int:

        """
        Return the number of currently stored entries.
        """
        
        return len(self._cache)
    


# Testing the code...


if __name__ == "__main__":

    import time

    print("=" * 60)
    print("Domain Cache Manual Test")
    print("=" * 60)

    # Use a short TTL so expiration can be tested quickly.
    cache = DomainCache[str](ttl_seconds=2)

    # ---------------------------------------------------------
    # Test 1: Cache starts empty
    # ---------------------------------------------------------
    print("\n[Test 1] Initial cache")
    print("Cache size:", cache.size())

    # ---------------------------------------------------------
    # Test 2: Store a value
    # ---------------------------------------------------------
    print("\n[Test 2] Store value")

    cache.set(
        "Example.COM.",
        "RDAP RESULT"
    )

    print("Cache size after set:", cache.size())

    # ---------------------------------------------------------
    # Test 3: Retrieve using normalized domain names
    # ---------------------------------------------------------
    print("\n[Test 3] Cache lookup")

    print(
        "example.com ->",
        cache.get("example.com")
    )

    print(
        "EXAMPLE.COM ->",
        cache.get("EXAMPLE.COM")
    )

    print(
        "Example.COM. ->",
        cache.get("Example.COM.")
    )

    # ---------------------------------------------------------
    # Test 4: Unknown domain
    # ---------------------------------------------------------
    print("\n[Test 4] Cache miss")

    print(
        "google.com ->",
        cache.get("google.com")
    )

    # ---------------------------------------------------------
    # Test 5: TTL expiration
    # ---------------------------------------------------------
    print("\n[Test 5] TTL expiration")

    print(
        "Before expiration ->",
        cache.get("example.com")
    )

    print("Waiting 3 seconds...")
    time.sleep(3)

    print(
        "After expiration ->",
        cache.get("example.com")
    )

    print(
        "Final cache size:",
        cache.size()
    )

    print("\n" + "=" * 60)
    print("Domain Cache Test Completed")
    print("=" * 60)