export function secondsToHHMMSS(seconds) {
    let date = new Date(seconds * 1000.0);
    return `${padSegment(date.getUTCHours())}:${padSegment(date.getUTCMinutes())}:${padSegment(date.getUTCSeconds())}`;
}

function padSegment(number) {
    return number.toString().padStart(2, "0");
}
