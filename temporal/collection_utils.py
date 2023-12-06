def get_first_element(coll, fallback = None):
    return next(iter(coll)) if coll else fallback

def reorder_dict(d, order):
    return {x: d[x] for x in order} | d
