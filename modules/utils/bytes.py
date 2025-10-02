from modules.utils.base64 import decode, decode_with_mime_type, encode, encode_with_mime_type


def base64_to_bytes(text: str, with_mime_type: bool = True) -> bytes:
    if with_mime_type:
        type, subtype, data = decode_with_mime_type(text)
    else:
        type, subtype, data = "application", "octet-stream", decode(text)

    if type != "application" or subtype != "octet-stream":
        raise ValueError

    return data


def bytes_to_base64(data: bytes, with_mime_type: bool = True) -> str:
    if with_mime_type:
        return encode_with_mime_type("application", "octet-stream", data)
    else:
        return encode(data)
