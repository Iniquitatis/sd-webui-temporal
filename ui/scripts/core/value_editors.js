import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";
import {colorToHex} from "../../scripts/utils/color.js";
import {createElement} from "../../scripts/utils/dom.js";

export class ValueEditor extends Widget {
    constructor() {
        super();

        this.style.width = "100%";

        this._header = createElement(this, "div", (e) => {
            e.style.alignItems = "center";
            e.style.display = "flex";
            e.style.justifyContent = "space-between";

            this._label = createElement(e, "label", (e) => {
                e.style.color = "var(--hint-color)";
                e.style.fontSize = "0.9rem";
                e.style.padding = "var(--vertical-padding) 0";
            });
        });

        this._content = createElement(this, "div", (e) => {
            e.style.display = "flex";
            e.style.flexDirection = "row";
            e.style.gap = "var(--layout-gap)";
        });
    }

    get label() {
        return this._label.innerText;
    }

    set label(value) {
        this._label.innerText = value;
    }

    createChild(cls, initializer, ...args) {
        return createElement(this._content, cls, initializer, ...args);
    }
}

export class BoolEditor extends ValueEditor {
    constructor() {
        super();

        this._input = createElement(this._header, "input", (e) => {
            e.type = "checkbox";
            e.style.marginLeft = "var(--horizontal-padding)";
            e.addEventListener("click", (event) => {
                event.stopPropagation();

                this.onValueChange.fire(e.checked);
            });
        });

        this.addEventListener("click", () => {
            this._input.checked = !this._input.checked;

            this.onValueChange.fire(this._input.checked);
        });

        this.onValueChange = new Signal();
    }

    get value() {
        return this._input.checked;
    }

    set value(value) {
        this._input.checked = value;

        this.onValueChange.fire(this.value);
    }

    createChild(cls, initializer, ...args) {
        return createElement(this._header, cls, initializer, ...args);
    }
}
customElements.define("bool-editor", BoolEditor);

export class NumberEditor extends ValueEditor {
    constructor() {
        super();

        // NOTE: Intentionally disconnected
        this._headerInput = createElement(null, "input", (e) => {
            e.type = "number";
            e.style.textAlign = "right";
            e.style.width = "var(--small-input-width)";
            e.addEventListener("input", () => {
                this._input.valueAsNumber = e.valueAsNumber;

                this.onValueChange.fire(e.valueAsNumber);
            });
        });

        this._input = createElement(this._content, "input", (e) => {
            e.type = "number";
            e.style.width = "100%";
            e.addEventListener("input", () => {
                this._headerInput.valueAsNumber = e.valueAsNumber;

                this.onValueChange.fire(e.valueAsNumber);
            });
        });

        this.onValueChange = new Signal();
    }

    get maximum() {
        return this._input.max;
    }

    get minimum() {
        return this._input.min;
    }

    get step() {
        return this._input.step;
    }

    get value() {
        return this._input.valueAsNumber;
    }

    get variant() {
        return this._input.type == "range" ? "slider" : "box";
    }

    set maximum(value) {
        this._input.max = value;
        this._headerInput.max = value;
    }

    set minimum(value) {
        this._input.min = value;
        this._headerInput.min = value;
    }

    set step(value) {
        this._input.step = value;
        this._headerInput.step = value;
    }

    set value(value) {
        this._input.valueAsNumber = value;

        if (this.variant == "slider") {
            this._headerInput.valueAsNumber = value;
        }

        this.onValueChange.fire(this.value);
    }

    set variant(value) {
        this._input.type = value == "slider" ? "range" : "number";

        if (value == "slider" && !this._header.contains(this._headerInput)) {
            this._header.appendChild(this._headerInput);
            this._input.style.marginTop = "var(--vertical-padding)";
        } else if (this._header.contains(this._headerInput)) {
            this._header.removeChild(this._headerInput);
            this._input.style.marginTop = undefined;
        }
    }
}
customElements.define("number-editor", NumberEditor);

export class TextEditor extends ValueEditor {
    constructor() {
        super();

        this._input = createElement(this._content, "input", (e) => {
            e.type = "text";
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(e.value);
            });
        });

        // NOTE: Intentionally disconnected
        this._textArea = createElement(null, "textarea", (e) => {
            e.rows = 5;
            e.style.display = "block";
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(e.value);
            });
        });

        this.onValueChange = new Signal();
    }

    get value() {
        return this._content.contains(this._textArea) ? this._textArea.value : this._input.value;
    }

    get variant() {
        return this._content.contains(this._textArea) ? "area" : "box";
    }

    set value(value) {
        if (this._content.contains(this._textArea)) {
            this._textArea.value = value;
        }
        else {
            this._input.value = value;
        }

        this.onValueChange.fire(this.value);
    }

    set variant(value) {
        if (value == "area" && this._content.contains(this._input) && !this._content.contains(this._textArea)) {
            this._content.removeChild(this._input);
            this._content.appendChild(this._textArea);
        } else if (this._content.contains(this._textArea) && !this._content.contains(this._input)) {
            this._content.removeChild(this._textArea);
            this._content.appendChild(this._input);
        }
    }
}
customElements.define("text-editor", TextEditor);

export class EnumEditor extends ValueEditor {
    constructor() {
        super();

        this._choices = {};

        this._select = createElement(this._content, "select", (e) => {
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(this.value);
            });
        });

        this.onValueChange = new Signal();
    }

    get choices() {
        return this._choices;
    }

    get value() {
        return [...Object.keys(this._choices)][this._select.selectedIndex];
    }

    set choices(value) {
        this._choices = value;

        while (this._select.contains(this._select.firstChild)) {
            this._select.removeChild(this._select.firstChild);
        }

        for (let name of Object.values(this._choices)) {
            createElement(this._select, "option", (e) => {
                e.label = name;
            });
        }
    }

    set value(value) {
        this._select.selectedIndex = [...Object.keys(this._choices)].indexOf(value);

        this.onValueChange.fire(this.value);
    }
}
customElements.define("enum-editor", EnumEditor);

class ChannelSlider extends ValueEditor {
    constructor() {
        super();

        let slider = createElement(this._header, "input", (e) => {
            e.type = "range";
            e.min = 0.0;
            e.max = 1.0;
            e.step = 0.01;
            e.valueAsNumber = 0.0;
            e.style.marginLeft = "var(--horizontal-padding)";
            e.style.marginRight = "calc(var(--horizontal-padding) / 2)";
            e.style.width = "100%";
            e.addEventListener("input", () => {
                box.valueAsNumber = e.valueAsNumber;

                this.onValueChange.fire(e.valueAsNumber);
            });
        });

        let box = createElement(this._header, "input", (e) => {
            e.type = "number";
            e.min = 0.0;
            e.max = 1.0;
            e.step = 0.01;
            e.valueAsNumber = 0.0;
            e.style.width = "var(--small-input-width)";
            e.addEventListener("input", () => {
                slider.valueAsNumber = e.valueAsNumber;

                this.onValueChange.fire(e.valueAsNumber);
            });
        });

        this.onValueChange = new Signal();
    }
}
customElements.define("channel-slider", ChannelSlider);

export class ColorEditor extends ValueEditor {
    constructor(channels) {
        super();

        createElement(this._content, "div", (e) => {
            e.style.display = "flex";
            e.style.flexDirection = "column";
            e.style.gap = "var(--layout-gap)";
            e.style.width = "100%";

            let color = {r: 0.0, g: 0.0, b: 0.0, a: 0.0};

            for (let channel of Object.keys(color).slice(0, channels)) {
                createElement(e, ChannelSlider, (e) => {
                    e.label = channel.toUpperCase();
                    e.onValueChange.connect((value) => {
                        color[channel] = value;

                        let hex = colorToHex(color);

                        this._preview.style.backgroundColor = hex;

                        this.onValueChange.fire(hex);
                    });
                });
            }
        });

        createElement(this._content, "div", (e) => {
            let gradient = "linear-gradient(" +
                "45deg, " +
                "hsl(0 0% 75%) 25%, " +
                "transparent 25%, " +
                "transparent 75%, " +
                "hsl(0 0% 75%) 75%)";

            e.style.background = `${gradient}, ${gradient}, hsl(0 0% 25%)`;
            e.style.backgroundPosition = "0 0, 8px 8px";
            e.style.backgroundRepeat = "repeat repeat";
            e.style.backgroundSize = "16px 16px";
            e.style.border = "var(--thin-border)";
            e.style.borderRadius = "var(--corners)";
            e.style.width = "var(--small-input-width)";

            this._preview = createElement(e, "div", (e) => {
                e.style.backgroundColor = "transparent";
                e.style.height = "100%";
                e.style.width = "100%";
            });
        });

        this.onValueChange = new Signal();
    }
}
customElements.define("color-editor", ColorEditor);

export class ImageEditor extends ValueEditor {
    constructor() {
        super();

        this._input = createElement(this._content, "input", (e) => {
            e.type = "file";
            e.style.minHeight = "10rem";
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(e.value);
            });
        });

        this._image = createElement(this._content, "div", (e) => {
            e.innerText = "IMAGE";
            e.style.alignContent = "center";
            e.style.border = "var(--thin-border)";
            e.style.borderRadius = "var(--corners)";
            e.style.color = "var(--hint-color)";
            e.style.fontSize = "1.6rem";
            e.style.height = "auto";
            e.style.textAlign = "center";
            e.style.userSelect = "none";
            e.style.width = "100%";
        });

        this.onValueChange = new Signal();
    }

    get value() {
        return this._input.value;
    }

    set value(value) {
        this._input.value = value;

        this.onValueChange.fire(this.value);
    }
}
customElements.define("image-editor", ImageEditor);
