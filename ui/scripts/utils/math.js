export function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

export function countFractionDigits(number) {
    let string = number.toString();
    let dotIndex = string.indexOf(".");
    return dotIndex != -1 ? string.length - dotIndex - 1 : 0;
}

export function lerp(a, b, x) {
    return a * (1.0 - x) + b * x;
}

export function normalize(value, min, max) {
    return (value - min) / (max - min);
}

export function quantize(value, step) {
    return Math.round(value / step) * step;
}
