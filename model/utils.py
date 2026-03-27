from ipaddress import ip_address

def is_ip_domain(domain):

    try:
        ip_address(domain)
        return 1                       # Successfully parsed it's an IP
    except ValueError:
        return 0                       # couldn't parse = it's a domain name 