import {createElement} from "../../scripts/utils/dom.js";

export class FileDownloader {
    constructor() {
        this._a = createElement(null, "a");
    }

    async download(value, fileName) {
        await fetch(value)
        .then((response) => response.blob())
        .then((blob) => {
            let mimeType = value.substring(value.indexOf(":") + 1, value.indexOf(";"));
            let format = mimeType.substring(mimeType.indexOf("/") + 1);

            this._a.download = `${fileName}.${format}`;
            this._a.href = URL.createObjectURL(blob);
            this._a.dataset.downloadurl = [mimeType, this._a.download, this._a.href];
            this._a.click();
        });
    }
}
