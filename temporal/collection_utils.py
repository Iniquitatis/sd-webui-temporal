def get_first_element(coll, fallback = None):
    return next(iter(coll)) if coll else fallback
