import {Block} from "/scripts/base/block.js";
import {Overlay} from "/scripts/base/overlay.js";
import {Row} from "/scripts/base/row.js";
import {ToolButton} from "/scripts/base/tool_button.js";
import {FileDownloader} from "/scripts/core/file_downloader.js";
import {FilePicker} from "/scripts/core/file_picker.js";
import {Signal} from "/scripts/core/signal.js";
import {createElement, defineElement} from "/scripts/utils/dom.js";

class MediaViewer extends Overlay {
    static tag = "ce-media-viewer";
    static css = `
        <self> > ce-block > .element {
            height: 100%;
        }
    `;

    constructor(element) {
        super();

        this._element = this._content.appendChild(element.cloneNode());
    }

    get value() {
        return this._element.value;
    }

    set value(value) {
        this._element.value = value;
    }
}
defineElement(MediaViewer);

export class MediaBox extends Block {
    static tag = "ce-media-box";
    static css = `
        <self> {
            align-content: center;
            background: var(--input-color);
            border: var(--thin-border);
            border-radius: var(--corners);
            height: 100%;
            min-height: 150px;
            overflow: hidden;
            position: relative;
            text-align: center;
            user-select: none;
        }

        <self> > .upload-background {
            align-content: center;
            color: var(--hint-color);
            font-size: 96px;
            inset: 0;
            opacity: 0.25;
            position: absolute;
        }

        <self> > .element {
            height: 100%;
        }

        <self> > .tools {
            flex-direction: row-reverse;
            gap: var(--layout-small-gap);
            position: absolute;
            right: var(--layout-padding);
            top: var(--layout-padding);
            width: auto;
        }
    `;

    constructor(cls, mimeType = "*/*", features = []) {
        super();

        this.onValueChange = new Signal();

        this._filePicker = new FilePicker();
        this._filePicker.mimeType = mimeType;
        this._filePicker.onLoad.connect((data) => {
            this.value = data;
        });

        this._fileDownloader = new FileDownloader();

        if (features.includes("upload")) {
            this.createChild(Block, (e) => {
                e.classList.add("upload-background");
                e.innerText = "\u{f093}";
                e.addEventListener("click", () => {
                    if (!this.value) {
                        this._filePicker.open();
                    }
                });
                this.onValueChange.connect((value) => {
                    e.visible = !value;
                });
            });
        }

        this._element = this.createChild(cls, (e) => {
            e.classList.add("element");
            e.visible = false;
            e.onValueChange.connect((value) => {
                if (this._viewer) {
                    this._viewer.value = value;
                }

                this.onValueChange.fire(this.value);
            });
            this.onValueChange.connect((value) => {
                e.visible = !!value;
            });
        });

        this._tools = this.createChild(Row, (e) => {
            e.classList.add("tools");
            e.visible = false;
            this.onValueChange.connect((value) => {
                e.visible = !!value;
            });
        });

        if (features.includes("clear")) {
            this.addTool("\u{f2ed}", () => {
                this.value = null;
            });
        }

        if (features.includes("fullscreen")) {
            this.addTool("\u{f065}", () => {
                this._viewer = createElement(document.body, MediaViewer, (e) => {
                    e.value = this.value;

                    if (features.includes("download")) {
                        e.addTool("\u{f019}", () => {
                            this._fileDownloader.download(this.value, getDownloadFileName());
                        });
                    }

                    e.onClose.connect(() => {
                        this._viewer = null;
                    });
                }, this._element);
            });
        }

        if (features.includes("upload")) {
            this.addTool("\u{f093}", () => {
                this._filePicker.open();
            });

            this.addEventListener("dragover", (event) => {
                event.preventDefault();

                if (this.classList.contains("drag-over")) return;

                if (![...event.dataTransfer.items].some((item) =>
                    (item.kind == "string" && matchMimeType(item.type, "text/plain")) ||
                    (item.kind == "file" && matchMimeType(item.type, mimeType))
                )) return;

                this.toggleClass("drag-over", true);
            });

            this.addEventListener("dragleave", (event) => {
                this.toggleClass("drag-over", false);
            });

            this.addEventListener("drop", (event) => {
                event.preventDefault();

                if (!this.classList.contains("drag-over")) return;

                let item = event.dataTransfer.items[0];

                if (item.kind == "string" && matchMimeType(item.type, "text/plain")) {
                    item.getAsString((data) => {
                        if (matchDataMimeType(data, mimeType)) this.value = data;
                    });
                } else if (item.kind == "file" && matchMimeType(item.type, mimeType)) {
                    let file = item.getAsFile();

                    let reader = new FileReader();
                    reader.addEventListener("load", (event) => {
                        this.value = event.target.result;
                    });
                    reader.readAsDataURL(file);
                }

                this.toggleClass("drag-over", false);
            });
        }

        if (features.includes("download")) {
            this.addTool("\u{f019}", () => {
                this._fileDownloader.download(this.value, getDownloadFileName());
            });
        }

        this._viewer = null;
    }

    get height() {
        return this.style.minHeight;
    }

    get value() {
        return this._element.value;
    }

    set height(value) {
        this.style.minHeight = value;
    }

    set value(value) {
        this._element.value = value;
    }

    addTool(icon, callback) {
        this._tools.createChild(ToolButton, (e) => {
            e.label = icon;
            e.onClick.connect(callback);
        });
    }
}
defineElement(MediaBox);

function getDownloadFileName() {
    return `temporal_${new Date(Date.now()).toISOString()}`;
}

function matchDataMimeType(data, mask) {
    let [type, subtype] = mask.split("/");

    return data.startsWith(`data:${type}/${subtype != "*" ? subtype : ""}`);
}

function matchMimeType(target, mask) {
    let [targetType, targetSubtype] = target.split("/");
    let [maskType, maskSubtype] = mask.split("/");

    return (maskType == "*" || maskType == targetType) &&
        (maskSubtype == "*" || maskSubtype == targetSubtype);
}
