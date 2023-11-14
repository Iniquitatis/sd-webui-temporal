from copy import copy

def copy_with_overrides(obj, **overrides):
    instance = copy(obj)

    for key, value in overrides.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
        else:
            print(f"WARNING: Key {key} doesn't exist in {instance.__class__.__name__}")

    return instance
