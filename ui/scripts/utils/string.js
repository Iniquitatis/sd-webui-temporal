export function dedent(text) {
    let lines = text.split("\n");

    let startLine = -1;
    let endLine = -1;
    let baseLevel = Infinity;

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];

        let trimmed = line.trimStart();
        if (trimmed == "") continue;

        if (startLine == -1) startLine = i;
        endLine = i;

        let indent = line.length - trimmed.length;
        if (indent < baseLevel) baseLevel = indent;
    }

    if (startLine == -1) return "";

    return lines
        .slice(startLine, endLine + 1)
        .map((x) => x.slice(baseLevel))
        .join("\n");
}
