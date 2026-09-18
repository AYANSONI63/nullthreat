from __future__ import annotations

from pydantic import BaseModel

from ..cache.domain_cache import DomainCache

from .domain_registration import (
    DomainRegistration,
    extract_hostname,
    extract_registered_domain,
    get_domain_registration,
)

from .dns_intelligence import (
    DNSIntelligence,
    get_dns_intelligence,
)


# ===============================================================
# Configuration
# ===============================================================

DEFAULT_REGISTRATION_CACHE_TTL = 86400   # 24 hr
DEFAULT_DNS_CACHE_TTL = 3600        #1 hr   


# ===============================================================
# Pydantic Output Model
# ===============================================================

class DomainEnrichment(BaseModel):
    """
    Unified domain intelligence produced by
    NullThreat's enrichment pipeline.
    """

    domain: str

    registration: DomainRegistration | None = None

    dns: DNSIntelligence | None = None


# ===============================================================
# Enrichment Service
# ===============================================================

class EnrichmentService:
    """
    Coordinates the different domain-enrichment components.

    Responsibilities:

        1. Normalize the domain.
        2. Check the RDAP cache.
        3. Perform RDAP lookup when necessary.
        4. Store RDAP results in the cache.
        5. Perform DNS intelligence lookup.
        6. Combine all results.
    """

    def __init__(
        self,
        registration_cache: DomainCache[DomainRegistration] | None = None,
        dns_cache: DomainCache[DNSIntelligence] | None=None,
        registration_cache_ttl: int = DEFAULT_REGISTRATION_CACHE_TTL,
        dns_cache_ttl:  int=DEFAULT_DNS_CACHE_TTL,
    ) -> None:

        # --------------------------------------------------------
        # Registration cache
        # --------------------------------------------------------

        self._registration_cache = (
            registration_cache
            if registration_cache is not None
            else DomainCache[DomainRegistration](
                ttl_seconds=registration_cache_ttl
            )
        )

        # --------------------------------------------------------
        # DNS cache
        # --------------------------------------------------------
        
        self._dns_cache = (
            dns_cache
            if dns_cache is not None
            else DomainCache[DNSIntelligence](
                ttl_seconds=dns_cache_ttl
            )
        )


    # ===========================================================
    # Main Enrichment operation 
    # ===========================================================

    def enrich(
        self,
        value: str,
    ) -> DomainEnrichment:

        # -------------------------------------------------------
        # 1. Extract hostname
        # -------------------------------------------------------

        hostname = extract_hostname(
            value
        )

        # -------------------------------------------------------
        # 2. Extract registered domain
        # -------------------------------------------------------

        domain = extract_registered_domain(
            hostname
        )

        # -------------------------------------------------------
        # 3. Check RDAP cache
        # -------------------------------------------------------

        registration = self._registration_cache.get(
            domain
        )

        # -------------------------------------------------------
        # 4. Cache MISS → perform RDAP lookup
        # -------------------------------------------------------

        if registration is None:

            registration = (
                get_domain_registration(
                    domain
                )
            )

            # Store the RDAP result.
            self._registration_cache.set(
                domain,
                registration
            )

        # -------------------------------------------------------
        # 5. DNS lookup
        # -------------------------------------------------------

        dns_information = self._dns_cache.get(domain)
        

        if dns_information is None:
            dns_information = get_dns_intelligence(domain)

            self._dns_cache.set(
                domain,
                dns_information,
            )

        # -------------------------------------------------------
        # 6. Combine everything
        # -------------------------------------------------------

        return DomainEnrichment(
            domain=domain,
            registration=registration,
            dns=dns_information,
        )


# ===============================================================
# Manual Integration Test
# ===============================================================

if __name__ == "__main__":

    registration_cache = DomainCache[DomainRegistration](
        ttl_seconds=60
    )

    dns_cache = DomainCache[DNSIntelligence](
        ttl_seconds=60
    )

    service = EnrichmentService(
        registration_cache=registration_cache,
        dns_cache=dns_cache,
    )

    test_values = [
        "https://www.google.com",
        "https://google.com",
        "https://www.google.com",
    ]

    for value in test_values:

        print("\n" + "=" * 60)
        print(f"Enriching: {value}")
        print("=" * 60)

        result = service.enrich(value)

        print(
            result.model_dump_json(
                indent=4
            )
        )

        print(
            "\nRegistration cache size:",
            registration_cache.size(),
        )

        print(
            "DNS cache size:",
            dns_cache.size(),
        )