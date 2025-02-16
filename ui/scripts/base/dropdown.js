import {Signal} from "../../scripts/core/signal.js";
import {StateManager} from "../../scripts/core/state_manager.js";
import {Widget} from "../../scripts/core/widget.js";
import {clearElement, createElement} from "../../scripts/utils/dom.js";

export class Dropdown extends Widget {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._stateManager = new StateManager();
        this._stateManager.onStatesChange.connect((_states, _names, data) => {
            clearElement(this._select);

            for (let name of data) {
                createElement(this._select, "option", (e) => {
                    e.label = name;
                });
            }

            this._select.selectedIndex = this._stateManager.index;
        });
        this._stateManager.onValueChange.connect((value, _) => {
            this._select.selectedIndex = this._stateManager.index;

            this.onValueChange.fire(value);
        });

        this._select = this.createChild("select", (e) => {
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this._stateManager.index = e.selectedIndex;
            });
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
customElements.define("custom-dropdown", Dropdown);
