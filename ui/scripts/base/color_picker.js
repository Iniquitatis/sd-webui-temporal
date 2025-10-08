import {Block} from "/scripts/base/block.js";
import {Column} from "/scripts/base/column.js";
import {Row} from "/scripts/base/row.js";
import {Slider} from "/scripts/base/slider.js";
import {FieldManager} from "/scripts/core/field_manager.js";
import {Signal} from "/scripts/core/signal.js";
import {colorToHex} from "/scripts/utils/color.js";
import {defineElement} from "/scripts/utils/dom.js";

export class ColorPicker extends Row {
    static tag = "ce-color-picker";
    static css = `
        <self> > ce-column {
            gap: var(--layout-small-gap);
        }

        <self> > ce-column > ce-row > ce-block {
            align-content: center;
            color: var(--hint-color);
            font-size: 0.9rem;
            height: var(--widget-height);
            width: 1rem;
        }

        <self> > ce-column > ce-row > ce-slider {
            width: 100%;
        }

        <self> > ce-column > ce-row:nth-of-type(1) > ce-slider > .fill {
            background: oklch(from var(--fill-color) l 25% 30deg);
        }

        <self> > ce-column > ce-row:nth-of-type(2) > ce-slider > .fill {
            background: oklch(from var(--fill-color) l 25% 150deg);
        }

        <self> > ce-column > ce-row:nth-of-type(3) > ce-slider > .fill {
            background: oklch(from var(--fill-color) l 25% 270deg);
        }

        <self> > ce-column > ce-row:nth-of-type(4) > ce-slider > .fill {
            background: oklch(from var(--fill-color) l 0% 0deg);
        }

        <self> > ce-block {
            border: var(--thin-border);
            border-radius: var(--corners);
            overflow: hidden;
            width: var(--small-input-width);
        }

        <self> > ce-block > ce-block {
            height: 100%;
            width: 100%;
        }
    `;

    constructor(channels) {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager._value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};

        this.createChild(Column, (e) => {
            for (let channel of Object.keys(this._manager._value).slice(0, channels)) {
                e.createChild(Row, (e) => {
                    e.createChild(Block, (e) => {
                        e.innerText = channel.toUpperCase();
                    });

                    e.createChild(Slider, (e) => {
                        e.minimum = 0.0;
                        e.maximum = 1.0;
                        e.step = 0.01;
                        e.value = 0.0;
                        this._manager.manage(e, channel);
                    });
                });
            }
        });

        this.createChild(Block, (e) => {
            e.classList.add("checkerboard-bg");

            this._preview = e.createChild(Block, (e) => {
                e.style.backgroundColor = "transparent";
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

    isComplexWidget() {
        return true;
    }
}
defineElement(ColorPicker);
