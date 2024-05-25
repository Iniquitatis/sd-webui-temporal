import re

def get_first_element(coll, fallback = None):
    return next(iter(coll)) if coll else fallback

def natural_sort(iterable):
    return sorted(iterable, key = lambda item: tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", item)))

def reorder_dict(d, order):
    return {x: d[x] for x in order} | d
