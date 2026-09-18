from __future__ import annotations

import time
from urllib.parse import urljoin ,urlparse

import requests
from pydantic import BaseModel 

from ..security.network_policy import (
    check_destination,
)

# ===================================================================
# Configuration
# ===================================================================

DEFAULT_TIME = 8

USER_AGENT = (
    "NullThreat/1.0"
    "(Real-Time Security Scanner)"
)

MAX_REDIRECTS = 5 

REDIRECT_STATUS_CODES = {
    301,
    302,
    303,
    307,
    308,
}


# ===================================================================
# Exceptions
# ===================================================================

class HTTPInspectionError(Exception):
    """Base exception for HTTP inspection."""

class HTTPConnectionError(Exception):
    """Raised when the target cannot be reached."""

class HTTPTimeoutError(Exception):
    """Raised when the HTTP request times out."""

class HTTPInvalidURLError(Exception):
    """Raised when the supplied URL is invalid."""


# ===================================================================
# Pydantic Output Model 
# ===================================================================

class HTTPInspection(BaseModel):
    """
    Normalized HTTP intelligence collected by 
    NullThreat's real-time inspection layer.
    """

    url: str

    request_successful: bool = False 

    status_code: int | None = None 

    response_url: str | None = None 

    redirect_count: int = 0
    
    redirect_location: str | None = None

    redirect_blocked: bool = False

    redirect_block_reason: str | None = None

    content_type: str | None = None 

    server: str | None = None 

    content_length: int | None = None

    response_time_ms: float | None = None 


# ===================================================================
# URL Validation 
# ===================================================================


def validate_url(url: str) -> str:
    """
    Validate and normalize the URL.

    Only HTTP and HTTPS are allowed.
    """

    url = url.strip()

    if not url:
        raise HTTPInvalidURLError(
            "URL cannot be empty."
        )
    
    parsed = urlparse(url)


    if parsed.scheme.lower() not in {
        "http",
        "https",
    }:
        raise HTTPInvalidURLError(
            "Only HTTP and HTTPS URLs are allowed."
        )
    

    if not parsed.hostname:
        raise HTTPInvalidURLError(
            "URL does not contain a hostname."
        )

    return url


def resolve_redirect_url(
    original_url: str,
    location: str,
) -> str:
    """
    Convert a redirect Location header into 
    ans absolute URL.
    """

    return urljoin(
        original_url,
        location,
    )



def get_redirect_hostname(
    redirect_url: str,
) -> str:
    """
    Extract and validate the hostname from
    a redirect destination.
    """

    parsed = urlparse(redirect_url)

    if parsed.scheme.lower() not in {
        "http",
        "https",
    }:
        raise HTTPInvalidURLError(
            "Redirect destination must use "
            "HTTP or HTTPS."
        )

    if not parsed.hostname:
        raise HTTPInvalidURLError(
            "Redirect destination does not "
            "contain a hostname."
        )

    return parsed.hostname



# ====================================================================
# HTTP Inspection
# ====================================================================

def inspect_http(
    url: str,
    timeout: int = DEFAULT_TIME,
) -> HTTPInspection:
    """
    Perform a controlled HTTP request and collect
    HTTP-level intelligence.

    Redirects are inspected manually so every
    redirect destination can be validated
    through NullThreat's network security policy.
    """

    url = validate_url(url)

    start_time = time.perf_counter()

    current_url = url
    visited_urls = set()
    redirect_count = 0
    last_redirect_location = None

    while True:

        # --------------------------------------------------
        # Redirect loop protection
        # --------------------------------------------------

        if current_url in visited_urls:

            return HTTPInspection(
                url=url,
                request_successful=False,
                status_code=None,
                response_url=current_url,
                redirect_count=redirect_count,
                redirect_location=None,
                redirect_blocked=True,
                redirect_block_reason="redirect_loop",
                content_type=None,
                server=None,
                content_length=None,
                response_time_ms=round(
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    2,
                ),
            )

        visited_urls.add(current_url)

        # --------------------------------------------------
        # Validate current destination
        # --------------------------------------------------

        parsed = urlparse(current_url)

        hostname = parsed.hostname

        if not hostname:

            return HTTPInspection(
                url=url,
                request_successful=False,
                status_code=None,
                response_url=current_url,
                redirect_count=redirect_count,
                redirect_location=None,
                redirect_blocked=True,
                redirect_block_reason="invalid_hostname",
                content_type=None,
                server=None,
                content_length=None,
                response_time_ms=round(
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    2,
                ),
            )

        decision = check_destination(
            hostname
        )

        if not decision.allowed:

            return HTTPInspection(
                url=url,
                request_successful=False,
                status_code=None,
                response_url=current_url,
                redirect_count=redirect_count,
                redirect_location=None,
                redirect_blocked=True,
                redirect_block_reason=decision.reason,
                content_type=None,
                server=None,
                content_length=None,
                response_time_ms=round(
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    2,
                ),
            )

        # --------------------------------------------------
        # HTTP request
        # --------------------------------------------------

        try:

            response = requests.get(
                current_url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": (
                        "text/html,"
                        "application/xhtml+xml,"
                        "application/json,"
                        "*/*;q=0.8"
                    ),
                },
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )

        except requests.exceptions.Timeout as exc:

            raise HTTPTimeoutError(
                f"HTTP request timed out for "
                f"{current_url}: {exc}"
            ) from exc

        except requests.exceptions.ConnectionError as exc:

            raise HTTPConnectionError(
                f"HTTP connection failed for "
                f"{current_url}: {exc}"
            ) from exc

        except requests.exceptions.RequestException as exc:

            raise HTTPInspectionError(
                f"HTTP request failed for "
                f"{current_url}: {exc}"
            ) from exc

        # --------------------------------------------------
        # Response metadata
        # --------------------------------------------------

        content_length = None

        header_value = response.headers.get(
            "Content-Length"
        )

        if header_value:

            try:

                content_length = int(
                    header_value
                )

            except ValueError:

                content_length = None

        content_type = response.headers.get(
            "Content-Type"
        )

        server = response.headers.get(
            "Server"
        )

        # --------------------------------------------------
        # Normal response
        # --------------------------------------------------

        if response.status_code not in REDIRECT_STATUS_CODES:

            return HTTPInspection(
                url=url,
                request_successful=True,
                status_code=response.status_code,
                response_url=current_url,
                redirect_count=redirect_count,
                redirect_location=last_redirect_location,
                redirect_blocked=False,
                redirect_block_reason=None,
                content_type=content_type,
                server=server,
                content_length=content_length,
                response_time_ms=round(
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    2,
                ),
            )

        # --------------------------------------------------
        # Redirect detected
        # --------------------------------------------------

        location = response.headers.get(
            "Location"
        )

        if not location:

            return HTTPInspection(
                url=url,
                request_successful=False,
                status_code=response.status_code,
                response_url=current_url,
                redirect_count=redirect_count,
                redirect_location=None,
                redirect_blocked=True,
                redirect_block_reason=(
                    "redirect_without_location"
                ),
                content_type=content_type,
                server=server,
                content_length=content_length,
                response_time_ms=round(
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    2,
                ),
            )

        # --------------------------------------------------
        # Resolve redirect destination
        # --------------------------------------------------

        next_url = resolve_redirect_url(
            current_url,
            location,
        )

        last_redirect_location = next_url

        # Validate redirect URL itself
        next_url = validate_url(
            next_url
        )

        redirect_hostname = (
            get_redirect_hostname(
                next_url
            )
        )

        # --------------------------------------------------
        # Network security validation
        # --------------------------------------------------

        decision = check_destination(
            redirect_hostname
        )

        if not decision.allowed:

            return HTTPInspection(
                url=url,
                request_successful=False,
                status_code=response.status_code,
                response_url=current_url,
                redirect_count=redirect_count + 1,
                redirect_location=next_url,
                redirect_blocked=True,
                redirect_block_reason=(
                    decision.reason
                ),
                content_type=content_type,
                server=server,
                content_length=content_length,
                response_time_ms=round(
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    2,
                ),
            )

        # --------------------------------------------------
        # Maximum redirect protection
        # --------------------------------------------------

        redirect_count += 1

        if redirect_count >= MAX_REDIRECTS:

            return HTTPInspection(
                url=url,
                request_successful=False,
                status_code=response.status_code,
                response_url=current_url,
                redirect_count=redirect_count,
                redirect_location=next_url,
                redirect_blocked=True,
                redirect_block_reason=(
                    "max_redirects_exceeded"
                ),
                content_type=content_type,
                server=server,
                content_length=content_length,
                response_time_ms=round(
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    2,
                ),
            )

        # --------------------------------------------------
        # Continue to next redirect
        # --------------------------------------------------

        current_url = next_url    

# ===============================================================
# Manual Test
# ===============================================================

if __name__ == "__main__":

    test_urls = [
        "https://httpbin.org/redirect-to?url=https%3A%2F%2Fexample.com",
    ]


    for url in test_urls:

        print("=" * 60)

        print(
            f"Inspecting HTTP: {url}"
        )

        print("=" * 60)


        try:

            result = inspect_http(url)

            print(
                result.model_dump_json(
                    indent=4
                )
            )

        except HTTPInspectionError as exc:

            print(
                f"HTTP inspection failed: {exc}"
            )