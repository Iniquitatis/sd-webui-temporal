def safe_get_directory(path):
    path.mkdir(parents = True, exist_ok = True)
    return path
