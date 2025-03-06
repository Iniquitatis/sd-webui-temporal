import re
from io import BytesIO

from pybase64 import b64decode, b64encode


def decode(data: str) -> bytes:
    return b64decode(data, validate = True)


def encode(data: bytes) -> str:
    return b64encode(data).decode()


def decode_with_mime_type(text: str) -> tuple[str, str, bytes]:
    for (i, char), _ in zip(enumerate(text), range(128)):
        if char == ",":
            comma_pos = i
            break
    else:
        raise ValueError

    data_uri = text[:comma_pos]
    _, type, subtype, _ = re.split(r"[:;/]", data_uri)

    return (type, subtype, b64decode(text[comma_pos + 1:], validate = True))


def encode_with_mime_type(type: str, subtype: str, data: bytes) -> str:
    with BytesIO() as stream:
        stream.write(f"data:{type}/{subtype};base64,".encode())
        stream.write(b64encode(data))
        return stream.getvalue().decode()
