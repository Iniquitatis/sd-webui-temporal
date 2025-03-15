function channelToHex(channel) {
    let result = Math.round(channel * 255.0).toString(16);
    return result.length == 2 ? result : `0${result}`;
}

export function colorToHex(color, channels = 4) {
    return `#${channels >= 1 ? channelToHex(color.r ?? 0.0) : ""}` +
            `${channels >= 2 ? channelToHex(color.g ?? 0.0) : ""}` +
            `${channels >= 3 ? channelToHex(color.b ?? 0.0) : ""}` +
            `${channels >= 4 ? channelToHex(color.a ?? 1.0) : ""}`;
}

export function hexToColor(string) {
    let groups = string.match(/#([a-f\d]{2})([a-f\d]{2})?([a-f\d]{2})?([a-f\d]{2})?/i);

    return {
        r: groups[1] ? parseInt(groups[1], 16) / 255.0 : 0.0,
        g: groups[2] ? parseInt(groups[2], 16) / 255.0 : 0.0,
        b: groups[3] ? parseInt(groups[3], 16) / 255.0 : 0.0,
        a: groups[4] ? parseInt(groups[4], 16) / 255.0 : 1.0,
    };
}
