import {Block} from "../../scripts/base/block.js";
import {Column} from "../../scripts/base/column.js";
import {ValueEditor} from "../../scripts/base/value_editor.js";
import {FieldManager} from "../../scripts/core/field_manager.js";
import {Signal} from "../../scripts/core/signal.js";
import {colorToHex, hexToColor} from "../../scripts/utils/color.js";

class ChannelSlider extends ValueEditor {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._slider = this._header.createChild("input", (e) => {
            e.type = "range";
            e.min = 0.0;
            e.max = 1.0;
            e.step = 0.01;
            e.valueAsNumber = 0.0;
            e.style.marginLeft = "var(--horizontal-padding)";
            e.style.marginRight = "calc(var(--horizontal-padding) / 2)";
            e.style.width = "100%";
            e.addEventListener("input", () => {
                this._box.valueAsNumber = e.valueAsNumber;

                this.onValueChange.fire(e.valueAsNumber);
            });
        });

        this._box = this._header.createChild("input", (e) => {
            e.type = "number";
            e.min = 0.0;
            e.max = 1.0;
            e.step = 0.01;
            e.valueAsNumber = 0.0;
            e.style.width = "var(--small-input-width)";
            e.addEventListener("input", () => {
                this._slider.valueAsNumber = e.valueAsNumber;

                this.onValueChange.fire(e.valueAsNumber);
            });
        });
    }

    get value() {
        return this._box.value;
    }

    set value(value) {
        this._slider.value = parseFloat(value.toFixed(2));
        this._box.value = parseFloat(value.toFixed(2));
    }
}
customElements.define("channel-slider", ChannelSlider);

export class ColorPicker extends ValueEditor {
    constructor(channels) {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange, colorToHex, hexToColor);
        this._manager._value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};

        this._content.createChild(Column, (e) => {
            e.style.gap = "calc(var(--layout-gap) / 2)";

            for (let channel of Object.keys(this._manager._value).slice(0, channels)) {
                e.createChild(ChannelSlider, (e) => {
                    e.label = channel.toUpperCase();
                    this._manager.manage(e, channel);
                });
            }
        });

        this._content.createChild(Block, (e) => {
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

            this._preview = e.createChild(Block, (e) => {
                e.style.backgroundColor = "transparent";
                e.style.height = "100%";
                e.style.width = "100%";
                this.onValueChange.connect((value) => {
                    e.style.backgroundColor = value;
                });
            });
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("color-picker", ColorPicker);
