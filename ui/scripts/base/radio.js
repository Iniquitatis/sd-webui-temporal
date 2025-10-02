import {Block} from "/scripts/base/block.js";
import {Signal} from "/scripts/core/signal.js";
import {StateManager} from "/scripts/core/state_manager.js";
import {Widget} from "/scripts/core/widget.js";
import {clearElement} from "/scripts/utils/dom.js";

export class Radio extends Widget {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._stateManager = new StateManager();
        this._stateManager.onStatesChange.connect((states, _names, _data) => {
            clearElement(this._buttons);

            for (let [value, name] of Object.entries(states)) {
                let callback = (event) => {
                    this._stateManager.value = event.target.value;
                };

                this._buttons.createChild(Block, (e) => {
                    e.value = value;
                    e.style.alignItems = "center";
                    e.style.background = "var(--input-color)";
                    e.style.border = "var(--thin-border)";
                    e.style.borderRadius = "var(--corners)";
                    e.style.display = "flex";
                    e.style.flexDirection = "row";
                    e.style.gap = "calc(var(--horizontal-padding))";
                    e.style.height = "var(--widget-height)";
                    e.style.padding = "0 var(--horizontal-padding)";
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
            for (let button of this._buttons.childNodes) {
                button.input.checked = button.value == value;
            }

            this.onValueChange.fire(value);
        });

        this._buttons = this.createChild(Block, (e) => {
            e.style.display = "flex";
            e.style.flexWrap = "wrap";
            e.style.gap = "var(--layout-gap)";
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
customElements.define("custom-radio", Radio);
