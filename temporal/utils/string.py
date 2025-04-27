def ellipsize(string: str, limit: int) -> str:
    return string if len(string) <= limit else f"{string[:limit - 3]}..."
