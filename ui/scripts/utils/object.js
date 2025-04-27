export function deepCopy(object) {
    return JSON.parse(JSON.stringify(object));
}

export function mapValues(object, callback) {
    let result = {};

    for (let [key, value] of Object.entries(object)) {
        result[key] = callback(key, value);
    }

    return result;
}
