export function getObjectKeyByIndex(object, index) {
    return Object.keys(object)[index];
}

export function mapObject(object, callback) {
    let result = {};

    for (let key of Object.keys(object)) {
        result[key] = callback(key, object[key]);
    }

    return result;
}
