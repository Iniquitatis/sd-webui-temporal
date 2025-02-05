function channelToHex(channel) {
    let result = Math.round(channel * 255.0).toString(16);
    return result.length == 2 ? result : `0${result}`;
}

export function colorToHex(color) {
    return `#${channelToHex(color.r)}` +
            `${channelToHex(color.g)}` +
            `${channelToHex(color.b)}` +
            `${channelToHex(color.a)}`;
}
