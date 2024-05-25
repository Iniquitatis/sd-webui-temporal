import re

def match_mask(string: str, mask: str) -> bool:
    return bool(re.match(r"\b" + mask.replace("*", r".+?") + r"\b", string))
