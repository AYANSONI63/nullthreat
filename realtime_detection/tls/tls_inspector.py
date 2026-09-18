from __future__ import annotations

import socket
import ssl
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_TLS_PORT = 443
DEFAULT_TIMEOUT = 5.0



class TLSInformation(BaseModel):
    hostname: str

    tls_available: bool
    certificate_valid: bool | None = None

    tls_version: str | None = None
    cipher: str | None = None

    certificate_issuer: str | None = None
    certificate_subject: str | None = None


    certificate_not_before: str | None = None
    certificate_not_after: str | None = None 


    certificate_san: list[str] = Field(
        default_factory=list 
    )


def _extract_certificate_name(
    section: Any,
    preferred_names: tuple[str, ...],
) -> str | None:


    if not section:
        return None 

    for rnd in section:
        for key, value in rnd:
            if key in preferred_names:
                return value

    return None 


def _extract_san(
    certificate: dict[str, Any],
) -> list[str]:
    
    san_entries = certificate.get(
        "subjectAltName",
        (),
    )

    return  [
        value
        for entry_type, value in san_entries
        if entry_type == "DNS"
    ]



def inspect_tls(
    hostname: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> TLSInformation:
    
    hostname = (
        hostname
        .strip()
        .lower()
        .rstrip(".")
    )


    try:

        context = ssl.create_default_context()

        with socket.create_connection(
            (hostname, DEFAULT_TLS_PORT),
            timeout=timeout,
        ) as sock:
            
            with context.wrap_socket(
                sock,
                server_hostname=hostname,
            ) as ssock:

                certificate = ssock.getpeercert()

                cipher_info = ssock.cipher()

                return TLSInformation(
                    hostname=hostname,

                    tls_available=True,
                    certificate_valid=True,

                    tls_version=ssock.version(),

                    cipher=(
                        cipher_info[0]
                        if cipher_info
                        else None
                    ),

                    certificate_issuer=(
                        _extract_certificate_name(
                            certificate.get("issuer"),
                            (
                                "organizationName",
                                "commonName",
                            ),
                        )
                    ),

                    certificate_subject=(
                        _extract_certificate_name(
                            certificate.get("subject"),
                            (
                                "commonName",
                                "organizationName",
                            ),
                        )
                    ),

                    certificate_not_before=(
                        certificate.get(
                            "notBefore"
                        )
                    ),

                    certificate_not_after=(
                        certificate.get(
                            "notAfter"
                        )
                    ),

                    certificate_san=_extract_san(
                        certificate
                    ),
                )    
    
    except ssl.SSLCertVerificationError:

        return TLSInformation(
            hostname=hostname,
            tls_available=True,
            certificate_valid=False
        )

    except (
        socket.timeout,
        TimeoutError,
        socket.gaierror,
        ConnectionError,
        ssl.SSLError,
    ):
        
        return TLSInformation(
            hostname=hostname,
            tls_available=False,
        )




if __name__ == "__main__":

    test_hosts = [
        "google.com",
        "example.com",
        "bbc.co.uk",
    ]

    for hostname in test_hosts:

        print("\n" + "=" * 60)
        print(f"Inspecting TLS: {hostname}")
        print("=" * 60)

        result = inspect_tls(hostname)

        print(
            result.model_dump_json(
                indent=4
            )
        )