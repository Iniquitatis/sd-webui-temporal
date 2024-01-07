import re

def match_mask(string, mask):
    return bool(re.match(r"\b" + mask.replace("*", r".+?") + r"\b", string))
