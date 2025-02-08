import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {colorToHex} from "../../scripts/utils/color.js";
import {createElement} from "../../scripts/utils/dom.js";

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
