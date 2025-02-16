import {Block} from "../../scripts/base/block.js";
import {Column} from "../../scripts/base/column.js";
import {Row} from "../../scripts/base/row.js";
import {Slider} from "../../scripts/base/slider.js";
import {FieldManager} from "../../scripts/core/field_manager.js";
import {Signal} from "../../scripts/core/signal.js";
import {colorToHex} from "../../scripts/utils/color.js";

export class ColorPicker extends Row {
    constructor(channels) {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager._value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};

        this.createChild(Column, (e) => {
            e.style.gap = "calc(var(--layout-gap) / 2)";

            for (let channel of Object.keys(this._manager._value).slice(0, channels)) {
                e.createChild(Row, (e) => {
                    e.createChild(Block, (e) => {
                        e.innerText = channel.toUpperCase();
                        e.style.alignContent = "center";
                        e.style.color = "var(--hint-color)";
                        e.style.fontSize = "0.9rem";
                        e.style.height = "var(--widget-height)";
                        e.style.width = "1rem";
                    });

                    e.createChild(Slider, (e) => {
                        e.minimum = 0.0;
                        e.maximum = 1.0;
                        e.step = 0.01;
                        e.value = 0.0;
                        e.style.width = "100%";
                        this._manager.manage(e, channel);
                    });
                });
            }
        });

        this.createChild(Block, (e) => {
            e.classList.add("checkerboard-bg");
            e.style.border = "var(--thin-border)";
            e.style.borderRadius = "var(--corners)";
            e.style.width = "var(--small-input-width)";

            this._preview = e.createChild(Block, (e) => {
                e.style.backgroundColor = "transparent";
                e.style.height = "100%";
                e.style.width = "100%";
                this.onValueChange.connect((value) => {
                    e.style.backgroundColor = colorToHex(value);
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
