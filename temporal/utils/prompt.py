import re
from random import Random

from temporal.utils.collection import cartesian_product_at
from temporal.utils.math import clamp, product


def evaluate_prompt(prompt: str, iteration: int, seed: int = -1) -> str:
    def repl(m: re.Match[str]) -> str:
        if not (groups := re.findall(r"(\w+)(?:\s+(.+?))?\s*:\s*(.+)", m[1], flags = re.DOTALL)):
            raise ValueError

        parts = groups[0]

        func = parts[0]
        args = _clean_split(r"\s*,\s*", parts[1])
        variants = _clean_split(r"\s*\|\s*", parts[2])

        return _evaluate_func(func, args, variants, iteration, seed)

    return re.sub(r"\{(.+?)\}", repl, prompt, flags = re.DOTALL)


def _evaluate_func(func: str, args: list[str], variants: list[str], iteration: int, seed: int) -> str:
    if func == "switch":
        iterations, bounds = int(args[0]), args[1]

        return variants[_calc_index(bounds, iteration // iterations, len(variants))]

    elif func == "combine":
        mode, iterations, bounds = args[0], int(args[1]), args[2]

        variant_groups = [_clean_split(r"\s*\/\s*", x) for x in variants]
        total_combinations = product(len(x) for x in variant_groups)

        return ", ".join(cartesian_product_at(
            *variant_groups,
            index = _calc_index(bounds, iteration // iterations, total_combinations),
            major = mode == "major",
        ))

    elif func == "randomize":
        return Random(seed).choice(variants)

    else:
        raise NotImplementedError


def _calc_index(bounds: str, current: int, total: int) -> int:
    last = total - 1

    if bounds == "clamp":
        return clamp(current, 0, last)
    elif bounds == "repeat":
        return current % total
    elif bounds == "mirror":
        return last - abs(current % (last * 2) - last)
    else:
        raise ValueError


def _clean_split(separator: str, text: str) -> list[str]:
    return [x.strip() for x in re.split(separator, text)]
