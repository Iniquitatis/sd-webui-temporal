def lerp(a, b, x):
    return a * (1.0 - x) + b * x

def normalize(value, min, max):
    return (value - min) / (max - min)

def quantize(value, step, rounding_func = round):
    return rounding_func(value / step) * step

def remap_range(value, old_min, old_max, new_min, new_max):
    return new_min + (value - old_min) / (old_max - old_min) * (new_max - new_min)
