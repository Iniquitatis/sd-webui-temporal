import {Block} from "/scripts/base/block.js";
import {Signal} from "/scripts/core/signal.js";
import {StateManager} from "/scripts/core/state_manager.js";
import {Widget} from "/scripts/core/widget.js";
import {clearElement, defineElement} from "/scripts/utils/dom.js";

export class Radio extends Widget {
    static tag = "ce-radio";
    static css = `
        <self> {
            display: flex;
            flex-wrap: wrap;
            gap: var(--layout-gap);
        }

        <self> > ce-block {
            align-items: center;
            background: var(--input-color);
            border: var(--thin-border);
            border-radius: var(--corners);
            display: flex;
            flex-direction: row;
            gap: calc(var(--horizontal-padding));
            height: var(--widget-height);
            padding: 0 var(--horizontal-padding);
        }
    `;

    constructor() {
        super();

        this.onValueChange = new Signal();

        this._stateManager = new StateManager();
        this._stateManager.onStatesChange.connect((states, _names, _data) => {
            clearElement(this);

            for (let [value, name] of Object.entries(states)) {
                let callback = (event) => {
                    this._stateManager.value = event.target.value;
                };

                this.createChild(Block, (e) => {
                    e.value = value;
                    e.addEventListener("click", callback);

                    e.input = e.createChild("input", (e) => {
                        e.value = value;
                        e.type = "radio";
                        e.checked = this._stateManager.value == value;
                        e.addEventListener("click", callback);
                    });

                    e.createChild("label", (e) => {
                        e.value = value;
                        e.innerText = name;
                        e.addEventListener("click", callback);
                    });
                });
            }
        });
        this._stateManager.onValueChange.connect((value, _) => {
            for (let button of this.childNodes) {
                button.input.checked = button.value == value;
            }

            this.onValueChange.fire(value);
        });
    }

    get choices() {
        return this._stateManager.states;
    }

    get value() {
        return this._stateManager.value;
    }

    set choices(value) {
        this._stateManager.states = value;
    }

    set value(value) {
        this._stateManager.value = value;
    }
}
defineElement(Radio);
