export function getObjectKeyByIndex(object, index) {
    return Object.keys(object)[index];
}

export function mapObject(object, callback) {
    let result = {};

    for (let [key, value] of Object.entries(object)) {
        result[key] = callback(key, value);
    }

    return result;
}
