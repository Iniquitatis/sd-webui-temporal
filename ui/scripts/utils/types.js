export function boolToString(value) {
    return value ? "true" : "false";
}

export function stringToBool(value) {
    let lc = value.trim().toLowerCase();
    return lc == "true" || lc == "on" || lc == "1";
}

export function* iterateAncestry(cls, stopAt = null) {
    if (!cls || cls == stopAt) return;
    yield* iterateAncestry(Object.getPrototypeOf(cls), stopAt);
    yield cls;
}
