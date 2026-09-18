from __future__ import annotations

import logging 
import time
from typing import Any

import dns.exception
import dns.resolver
from pydantic import BaseModel, Field


# ===============================================================
# Configuration
# ===============================================================

DEFAULT_DNS_TIMEOUT = 2.0
DEFAULT_DNS_LIFETIME = 4.0

DEFAULT_RETRIES = 2 

# Public fallback resolvers.
# These are only used when the system resolver fails
FALLBACK_NAMESERVERS = [
    "1.1.1.1",       #Cloudflare   
    "8.8.8.8",       #Google   
]


# ===============================================================
# Logging
# ===============================================================

logger = logging.getLogger(__name__)



# ===============================================================
# Exceptions
# ===============================================================


class DNSError(Exception):
    """Base exception for DNS-related errors."""


class DNSUnavailableError(DNSError):
    """Raised when DNS infrastructure is unavailable."""


# ===============================================================
# Pydantic Output Model
# ===============================================================


class DNSIntelligence(BaseModel):
    """
    Normalized DNS intelligence used by
    NullThreat's real-time enrichment pipeline.
    """

    domain: str

    a_records: list[str] = Field(
        default_factory=list
    )

    aaaa_records: list[str] = Field(
        default_factory=list
    )

    cname: str | None = None


    mx_records: list[dict[str, Any]] = Field(
        default_factory=list
    )

    nameservers: list[str] = Field(
        default_factory=list
    )

    txt_records: list[str] = Field(
        default_factory=list
    )


# =================================================================
# Resolver Creation 
# =================================================================

def _create_resolver(
    nameservers: list[str] | None = None,
) -> dns.resolver.Resolver:
    
    """
    Create a configured DNS resolver.

    If nameservers is None, the system-configured
    DNS resolver is used.
    """

    resolver = dns.resolver.Resolver()

    resolver.timeout = DEFAULT_DNS_TIMEOUT
    resolver.lifetime = DEFAULT_DNS_LIFETIME
    
    if nameservers is not None:
        resolver.nameservers = nameservers 


    return resolver

# =================================================================
# DNS Query Helper 
# =================================================================

def _resolver_with_retry(
    domain: str,
    record_type: str,
    resolver: dns.resolver.Resolver,
    retries: int = DEFAULT_RETRIES,
):
    """
    Resolve one DNS record with retry support.

    Returns:
        dns.resolver.Answer

    Raises:
        dns.resolver.NoAnswer
        dns.resolver.NXDOMAIN
        dns.resolver.NoNameservers
        dns.exception.Timeout
        dns.exception.DNSException
    """

    last_exception: Exception | None = None

    for attempt in range(retries + 1):

        try:

            return resolver.resolve(
                domain,
                record_type,
            )
        
        except (
            dns.resolver.NoAnswer,
            dns.resolver.NXDOMAIN
        ):
            raise
        
        except (
            dns.resolver.NoNameservers,
            dns.exception.Timeout,
        ) as exc:
            
            last_exception = exc

            logger.warning(
                "DNS %s lookup attempt %d/%d failed for %s: %s",
                record_type,
                attempt + 1,
                retries + 1,
                domain,
                exc,
            )

            if attempt < retries:
                time.sleep(
                    0.25 * (attempt + 1)
                )
        
        except dns.exception.DNSException as exc:

            last_exception = exc

            logger.warning(
                "DNS %s lookup attempt %d/%d failed for %s: %s",
                record_type,
                attempt + 1,
                retries + 1,
                domain,
                exc,
            )


            if attempt < retries:
                time.sleep(
                    0.25 * (attempt + 1)
                )

    if last_exception is not None:
        raise last_exception
    
    raise DNSUnavailableError(
        f"Unable to resolve {record_type} for {domain}"
    )

# =================================================================
# Resilient DNS Query
# =================================================================

def _resilient_resolve(
    domain: str,
    record_type: str,
):
    """
    Perform a DNS lookup using:

        1. System resolver
        2. Retry
        3. Public fallback resolvers

    Returns:
        DNS answer.

    Raises:
        NoAnswer / NXDOMAIN when the record is genuinely absent.
        DNSException when all resolvers fail.
    """

    # -------------------------------------------------------------
    # 1. Primary system resolver
    # -------------------------------------------------------------

    primary_resolver = _create_resolver()

    try:

        return _resolver_with_retry(
            domain,
            record_type,
            primary_resolver,
        )
    
    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
    ):
        raise

    except dns.exception.DNSException as primary_error:

        logger.warning(
            "Primary DNS resolver failed for %s (%s): %s",
            domain,
            record_type,
            primary_error,
        )

    # ---------------------------------------------------------
    # 2. Fallback public resolver
    # ---------------------------------------------------------

    fallback_resolver = _create_resolver(
        FALLBACK_NAMESERVERS
    )

    try:

        return _resolver_with_retry(
            domain,
            record_type,
            fallback_resolver,
        )
    
    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
    ):
        raise

    except dns.exception.DNSException as fallback_error:

        raise DNSUnavailableError(
            f"DNS lookup failed for {record_type}"
            f"for {domain}: {fallback_error}"
        ) from fallback_error

# =================================================================
# A Records
# =================================================================

def resolve_a_records(
    domain: str,
) -> list[str]:
    """
    Resolve A records for a domain.

    Returns:
        A list of IPv4 addresses.

    Raises:
        DNSUnavailableError:
        When the DNS lookup cannot be completed.
    """


    try:

        answer = _resilient_resolve(
            domain,
            "A",
        )

        return [
            answer_item.address
            for answer_item in answer
        ]
    

    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
    ):
        return []

    except DNSUnavailableError:

        logger.warning(
            "A lookup unavailable for %s",
            domain,
        )

        return []


# =================================================================
# AAAA Records
# =================================================================


def resolve_aaaa_records(
    domain: str,
) -> list[str]:
    
    """
    Resolve AAAA records for a domain.

    Returns:
        A list of IPv6 addresses.

    Raises:
        DNSUnavailableError:
            When the DNS lookup cannot be completed.
    """


    try:

        answer = _resilient_resolve(
            domain,
            "AAAA",
        )

        return [
            answer_item.address
            for answer_item in answer
        ]

    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
    ):
        return []

    except DNSUnavailableError:

        logger.warning(
            "AAAA lookup unavailable for %s",
            domain,
        )

        return []

# ===============================================================
# CNAME
# ===============================================================

def resolve_cname(
    domain: str,
) -> str | None:
    """
    Resolve the CNAME record for a domain.

    Returns:
        The canonical hostname if a CNAME exists.
        None if the domain has no CNAME record.

    Raises:
        DNSUnavailableError:
            When the DNS lookup cannot be completed.
    """

    try:

        answer = _resilient_resolve(
            domain,
            "CNAME",
        )

        return (
            answer[0]
            .target
            .to_text()
            .rstrip(".")
        )

    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
    ):
        return None

    except DNSUnavailableError:

        logger.warning(
            "CNAME lookup unavailable for %s",
            domain,
        )

        return None

# ===============================================================
# MX Records
# ===============================================================

def get_mx_records(
    domain: str,
) -> list[dict[str, Any]]:

    """
    Resolve MX records for a domain.

    Each MX record contains:
        - preference
        - exchange/mail server
    """

    try:

        answers = _resilient_resolve(
            domain,
            "MX",
        )

        mx_records: list[
            dict[str, Any]
        ] = []

        for answer in answers:

            exchange = (
                str(answer.exchange)
                .rstrip(".")
                .lower()
            )

            # "." is a legitimate Null MX value.
            if exchange == "":
                exchange = "."

            mx_records.append(
                {
                    "preference": int(
                        answer.preference
                    ),
                    "exchange": exchange,
                }
            )

        mx_records.sort(
            key=lambda record:
            record["preference"]
        )

        return mx_records

    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
    ):
        return []

    except DNSUnavailableError:

        logger.warning(
            "MX lookup unavailable for %s",
            domain,
        )

        return []


# ===============================================================
# Nameservers
# ===============================================================

def get_nameservers(
    domain: str,
) -> list[str]:
    
    """
    Resolve authoritative nameservers for a domain
    """

    try:

        answers = _resilient_resolve(
            domain,
            "NS",
        )

        nameservers: list[str] = []

        for answer in answers:

            nameserver = (
                str(answer)
                .rstrip(".")
                .lower()
            )

            if nameserver:
                nameservers.append(
                    nameserver
                )

        return nameservers

    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
    ):
        return []

    except DNSUnavailableError:

        logger.warning(
            "NS lookup unavailable for %s",
            domain,
        )

        return []


# ===============================================================
# TXT Records
# ===============================================================

def get_txt_records(
    domain: str,
) -> list[str]:
    
    """
    Resolve TXT records for a domain.
    """


    try:

        answers = _resilient_resolve(
            domain,
            "TXT",
        )

        txt_records: list[str] = []

        for answer in answers:

            chunks = getattr(
                answer,
                "strings",
                [],
            )

            record = "".join(
                chunk.decode(
                    "utf-8",
                    errors="replace",
                )
                if isinstance(chunk, bytes)
                else str(chunk)
                for chunk in chunks
            )

            if record:
                txt_records.append(
                    record
                )

        return txt_records

    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
    ):
        return []

    except DNSUnavailableError:

        logger.warning(
            "TXT lookup unavailable for %s",
            domain,
        )

        return []


# ===============================================================
# Unified DNS Intelligence
# ===============================================================

def get_dns_intelligence(
    domain: str,
) -> DNSIntelligence:

    """
    Collect all supported DNS intelligence.

    Each DNS record is isolated from the others.
    A failure in one record does not terminate
    the complete DNS enrichment operation.
    """

    return DNSIntelligence(
        domain=domain,

        a_records=resolve_a_records(
            domain
        ),

        aaaa_records=resolve_aaaa_records(
            domain
        ),

        cname=resolve_cname(
            domain
        ),

        mx_records=get_mx_records(
            domain
        ),

        nameservers=get_nameservers(
            domain
        ),

        txt_records=get_txt_records(
            domain
        ),
    )


# ===============================================================
# Manual Test
# ===============================================================

if __name__ == "__main__":

    test_domains = [
        "example.com",
        "google.com",
        "bbc.co.uk",
        "flipkart.com",
    ]

    print("=" * 60)
    print("DNS INTELLIGENCE MANUAL TEST")
    print("=" * 60)

    for domain in test_domains:

        print()
        print("=" * 60)
        print(f"Testing : {domain}")
        print("=" * 60)

        try:

            result = get_dns_intelligence(
                domain
            )

            print(
                result.model_dump_json(
                    indent=4
                )
            )

        except Exception as exc:

            print(
                f"Unexpected DNS failure: {exc}"
            )
