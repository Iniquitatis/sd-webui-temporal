from base64 import b64decode, b64encode


def base64_to_bytes(data: str) -> bytes:
    return b64decode(data)


def bytes_to_base64(data: bytes) -> str:
    return b64encode(data).decode()
