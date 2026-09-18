from __future__ import annotations

import socket 

from pydantic import BaseModel
from ipaddress import ip_address



UNSAFE_DESTINATION_TYPES = {
    "loopback",
    "private",
    "link_local",
    "multicast",
    "unspecified",
    "reserved",
}



class NetworkDecision(BaseModel):
    allowed: bool
    reason: str | None = None
    destination_type: str



def classify_ip(address: str) -> str:

    ip = ip_address(address)

    if ip.is_loopback:
        return "loopback"

    if ip.is_link_local:
        return "link_local"

    if ip.is_unspecified:
        return "unspecified"

    if ip.is_multicast:
        return "multicast"

    if ip.is_private:
        return "private"

    if ip.is_reserved and not ip.is_global:
        return "reserved"

    return "public"


def resolve_hostname(hostname: str) -> list[str]:
    results = socket.getaddrinfo(
        hostname,
        None,
        type=socket.SOCK_STREAM,    
    )

    addresses = {
        result[4][0]
        for result in results
    }


    return sorted(addresses)


def check_destination(value: str) -> NetworkDecision:
    value = value.strip()

    # Direct IP address
    if is_ip_address(value):
        destination_type = classify_ip(value)

        if destination_type in UNSAFE_DESTINATION_TYPES:
            return NetworkDecision(
                allowed=False,
                reason=f"{destination_type}_ip",
                destination_type=destination_type,
            )

        return NetworkDecision(
            allowed=True,
            destination_type="public",
        )

    # Hostname
    addresses = resolve_hostname(value)

    if not addresses:
        return NetworkDecision(
            allowed=False,
            reason="dns_resolution_failed",
            destination_type="unresolved",
        )

    for address in addresses:
        destination_type = classify_ip(address)

        print(
            f"{address:<40} -> {destination_type}"
        )

        if destination_type in UNSAFE_DESTINATION_TYPES:
            return NetworkDecision(
                allowed=False,
                reason=f"{destination_type}_ip",
                destination_type=destination_type,
            )

    return NetworkDecision(
        allowed=True,
        destination_type="public",
    )



def is_ip_address(value: str) -> bool:
    try:
        ip_address(value)
        return True
    except ValueError:
        return False



if __name__ == "__main__":
    test_hosts = [
        "google.com",
        "example.com",
        "bbc.co.uk",
    ]

    for hostname in test_hosts:
        print("\n" + "=" * 60)
        print(f"Checking: {hostname}")
        print("=" * 60)

        decision = check_destination(hostname)

        print(decision.model_dump_json(indent=4))


    # address = ip_address("64:ff9b::9765:4051")

    # print("address       :", address)
    # print("is_private    :", address.is_private)
    # print("is_reserved   :", address.is_reserved)
    # print("is_global     :", address.is_global)
    # print("is_link_local :", address.is_link_local)
    # print("is_loopback   :", address.is_loopback)