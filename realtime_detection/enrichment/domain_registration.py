from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field
import requests
import tldextract

# ===============================================================
# Configuration 
# ===============================================================


IANA_RDAP_BOOTSTRAP_URL = (
    "https://data.iana.org/rdap/dns.json"
)


DEFAULT_TIMEOUT = 10


# =================================================================
# Exceptions 
# =================================================================

class RDAPError(Exception):
    """Base exception for RDAP-related errors."""


class RDAPNotFoundError(RDAPError):
    """Raised when a domain is not found in RDAP."""


class RDAPUnavailableError(RDAPError):
    """Raised when RDAP service cannot be reached."""


# =================================================================
# # Pydantic Output Model 
# =================================================================

class DomainRegistration(BaseModel):
    """
    Normalized domain-registration information used by 
    Nullthreat's real time enrichment pipeline.
    """

    domain: str

    rdap_available: bool = False

    registration_date: datetime | None = None
    expiration_date: datetime | None = None
    last_changed_date: datetime | None = None
 
    domain_age_days: int | None = None
    registration_period_days: int | None = None
    days_until_expiration: int | None = None

    registrar: str | None = None

    statuses: list[str] = Field(
        default_factory=list
    ) 

    nameservers: list[str] = Field(
        default_factory=list
    )


# =================================================================
# URL/ Domain Processing 
# =================================================================


def extract_hostname(value: str) -> str:
    """
    Extract hostname from either a URL or hostname.

    Examples:
        https://www.example.com/login
            -> www.example.com

        www.example.com
            -> www.example.com

        example.com
            -> example.com
    """

    value = value.strip()
    
    if not value:
        raise ValueError(
            "URL/domain cannot be empty."
        )
    
    if "://" not in value:
        value = f"//{value}"
    
    parsed = urlparse(value)

    hostname = parsed.hostname

    if not hostname:
        raise ValueError(
            f"Could not extract hostname from: {value}"
        )
    
    return hostname.rstrip(".").lower()


def extract_registered_domain(
    hostname: str,
) -> str:
    """
    Extract the registered domain using the
    Public Suffix List.

    Examples:
        www.google.com
            -> google.com

        forums.bbc.co.uk
            -> bbc.co.uk

        shop.example.co.in
            -> example.co.in
    """

    hostname = hostname.strip().rstrip(".").lower()

    extracted = tldextract.extract(hostname)

    if not extracted.domain:
        raise ValueError(
            f"Unable to determine registered domain: "
            f"{hostname}"
        )
    
    if not extracted.suffix:
        raise ValueError(
            f"Hostname does not contain a recognized "
            f"public suffix: {hostname}"
        )
    
    return extracted.top_domain_under_public_suffix


# =================================================================
# IANA RDAP Bootstrap
# =================================================================

def fetch_bootstrap_registry(
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    
    """
    Fetch IANA's RDAP DNS bootstrap registry.
    """

    try:

        response = requests.get(
            IANA_RDAP_BOOTSTRAP_URL,
            timeout = timeout, 
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:

        raise  RDAPUnavailableError(
            "Unable to retrieve IANA RDAP "
            f"bootstrap registry: {exc}"
        ) from exc
    
    except ValueError as exc:

        raise RDAPUnavailableError(
            "IANA RDAP bootstrap returned "
            "invalid JSON."
        ) from exc
        
    
    if "services" not in data:

        raise RDAPUnavailableError(
            "Invalid IANA bootstrap response."
        )

    return data 


def find_rdap_base_urls(
    domain: str,
    bootstrap_data: dict[str, Any],
) -> list[str]:
    
    """
    Find authoratative RDAP base URLs for a domain.
    
    Performs label-wise longest matching.
    """

    domain = domain.lower().rstrip(".")

    domain_labels = domain.split(".")

    services = bootstrap_data.get(
        "services",
        [],
    )

    best_match_length = -1

    matching_urls: list[str] = []

    for service in services:

        if len(service) !=2:
            continue

        entries, urls = service

        for entry in entries:

            entry = entry.lower().rstrip(".")

            entry_labels = (
                entry.split(".")
                if entry
                else []
            )

            if len(entry_labels) > len(domain_labels):
                continue 

            if (
                domain_labels[
                    -len(entry_labels):
                ] 
                != entry_labels
            ):
                continue

            match_length = len(entry_labels)


            if match_length > best_match_length:
                
                best_match_length = match_length

                matching_urls = list(urls)

            elif match_length == best_match_length:

                matching_urls.extend(urls)

        
    if not matching_urls:

        raise RDAPUnavailableError(
            f"No RDAP service found for: {domain}"
        )
        
    https_urls = [
        url 
        for url in matching_urls
        if url.lower().startswith("https://")
    ]

    return https_urls or matching_urls


# ==============================================================
# RDAP Query
# ==============================================================

def query_rdap(
    domain: str,
    base_urls: list[str],
    timeout: int = DEFAULT_TIMEOUT,      
) -> dict[str, Any]:
    """
    Query an authoritative RDAP server.
    """

    last_error: Exception | None = None

    for base_url in base_urls:
        
        base_url = base_url.rstrip("/") + "/"

        rdap_url = (
            f"{base_url}domain/{domain}"
        )

        try:

            response = requests.get(
                rdap_url,
                headers={
                    "Accept": "application/rdap+json",
                    "User-Agent": (
                        "NullThreat/1.0"
                    ),
                },
                timeout=timeout
            )

            if response.status_code == 404:

                raise RDAPNotFoundError(
                    f"Domain not found: {domain}"
                )
            
            if response.status_code == 429:

                last_error = RDAPError(
                    "RDAP rate limit reached."
                )

                continue

            response.raise_for_status()

            data = response.json()

            if data.get(
                "objectClassName"
            ) != "domain":
                
                raise RDAPError(
                    "RDAP response is not"
                    "a domain object."
                )
            
            return data
        
        except RDAPNotFoundError:
            raise

        except(
            requests.RequestException,
            ValueError,
            RDAPError,
        ) as exc:
            
            last_error = exc


    raise RDAPUnavailableError(
        f"Unable to retrieve RDAP data "
        f"for {domain}. "
        f"Last error: {last_error}"
    )



# ===================================================================
# RDAP Event Parsing 
# ===================================================================


def get_event_date(
    rdap_date: dict[str, Any],
    event_action: str,
) -> datetime | None:
    

    """
    Extract a specific RDAP event date.

    Example:
        registration
        expiration 
        last changed
    """

    for event in rdap_date.get(
        "events",
        [],
    ):
        
        if (
            event.get("eventAction")
            != event_action
        ):
            continue 
        
        event_date = event.get(
            "eventDate"
        )

        if not event_date:
            continue

        try:

            return datetime.fromisoformat(
                event_date.replace(
                    "Z",
                    "+00:00",
                )
            )
        
        except ValueError:
            continue

    return None 


# ===============================================================
# Calculate domain age in days 
# ===============================================================

def calculate_domain_age_days(
    registration_date: datetime | None,
    reference_time: datetime | None=None
) -> int | None:
    
    """
    Calculate the age of a domain in days.

    Returns None when registration date is unavailable.
    """


    if registration_date is None:
        return None

    if reference_time is None:
        reference_time = datetime.now(
            timezone.utc
        )
    
    if (
        reference_time.tzinfo is None
        and registration_date is not None
    ):
        reference_time = reference_time.replace(
            tzinfo=timezone.utc
        )

    age = reference_time - registration_date


    return max(
        0,
        age.days
    )


# ===============================================================
# Calculate registration period days 
# ===============================================================

def calculate_registration_period_days(
    registration_date: datetime | None,
    expiration_date: datetime | None,
) -> int | None:
    """
    Calculate the number of days between registration
    and expiration.
    """

    if (
        registration_date is None
        or expiration_date is None
    ):
        return None

    period = (
        expiration_date
        - registration_date
    )

    return max(
        0,
        period.days,
    )

# ===============================================================
# Calculate days untile expiration 
# ===============================================================

def calculate_days_until_expiration(
    expiration_date: datetime | None,
    reference_time: datetime | None = None,
) -> int | None:
    """
    Calculate the number of days until domain expiration.

    Negative values mean the expiration date has already passed.
    """

    if expiration_date is None:
        return None

    if reference_time is None:
        reference_time = datetime.now(
            timezone.utc
        )

    if expiration_date.tzinfo is None:
        expiration_date = expiration_date.replace(
            tzinfo=timezone.utc
        )

    difference = (
        expiration_date
        - reference_time
    )

    return difference.days



# ===============================================================
# Registrar Parsing
# ===============================================================

def get_registrar(
    rdap_data: dict[str, Any],
) -> str | None:
    """
    Extract registrar name from RDAP entities.

    RDAP commonly represents entity contact information 
    using jCard/vCard structures.
    """
    for entity in rdap_data.get(
        "entities",
        [],
    ):
        roles = entity.get(
            "roles",
            [],
        )

        if "registrar" not in roles:
            continue

        vcard_array = entity.get(
            "vcardArray"
        )

        if (
            not vcard_array
            or len(vcard_array) < 2
        ):
            continue

        properties = vcard_array[1]

        for property_data in properties:

            if (
                len(property_data) >= 4
                and property_data[0] == "fn"
            ):
                
                return property_data[3]
            
    return None


# ==============================================================
# Nameserver Parsing
# ==============================================================


def get_nameservers(
    rdap_data: dict[str, Any],
) -> list[str]:
    
    """
    Extract nameserver hostnames.
    """

    nameservers: list[str] = []


    for nameserver in rdap_data.get(
        "nameservers",
        [],
    ):
        
        hostname = nameserver.get(
            "ldhName"
        )

        if hostname:

            nameservers.append(
                hostname.lower()
            )
        
    return nameservers


# =============================================================
# Normalize RDAP Response
# =============================================================

def normalize_rdap_response(
    domain: str,
    rdap_data: dict[str, Any],
) -> DomainRegistration:
    """
    Convert raw RDAP JSON into the validated
    NullThreat Pydantic model.
    """

    registration_date = get_event_date(
        rdap_data,
        "registration",
    )

    expiration_date = get_event_date(
        rdap_data,
        "expiration",
    )

    last_changed_date = get_event_date(
        rdap_data,
        "last changed",
    )

    domain_age_days = calculate_domain_age_days(
        registration_date
    )

    registration_period_days = (
        calculate_registration_period_days(
            registration_date,
            expiration_date,
        )
    )

    days_until_expiration = (
        calculate_days_until_expiration(
            expiration_date
        )
    )

    return DomainRegistration(
        domain=domain,

        rdap_available=True,

        registration_date=registration_date,

        expiration_date=expiration_date,

        last_changed_date=last_changed_date,

        domain_age_days=domain_age_days,

        registration_period_days=(
            registration_period_days
        ),

        days_until_expiration=(
            days_until_expiration
        ),

        registrar=get_registrar(
            rdap_data
        ),

        statuses=rdap_data.get(
            "status",
            [],
        ),

        nameservers=get_nameservers(
            rdap_data
        ),
    )



# ============================================================
# Public Interface
# ============================================================


def get_domain_registration(
    value: str,
    timeout: int = DEFAULT_TIMEOUT,   
) -> DomainRegistration:
    """
    Main public interface for NullThreat.

    Accepts:
        URL
        hostname

    Returns:
        DomainRegistration
    """

    hostname = extract_hostname(
        value
    )

    registered_domain = (
        extract_registered_domain(
            hostname
        )
    )

    bootstrap_data = (
        fetch_bootstrap_registry(
            timeout=timeout
        )
    )

    base_urls = find_rdap_base_urls(
        registered_domain,
        bootstrap_data,
    )


    rdap_data = query_rdap(
        registered_domain,
        base_urls,
        timeout=timeout,
    )


    return normalize_rdap_response(
        registered_domain,
        rdap_data,
    )



# ==========================================================
# Manual Test 
# ==========================================================


if __name__ == "__main__":

    test_domains = [
        "https://example.com",
        "https://www.google.com",
        "https://www.bbc.co.uk",
    ]

    for value in test_domains:

        print("=" * 60)
        print(f"Testing : {value}")
        print("=" * 60)

        try:

            result = (
                get_domain_registration(
                    value
                )
            )

            print(
                result.model_dump_json(
                    indent=4
                )
            )

        except (
            RDAPError,
            ValueError,
        ) as exc:

            print(f"RDAP lookup failed {exc}")


