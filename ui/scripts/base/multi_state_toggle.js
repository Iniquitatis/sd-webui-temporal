import {Signal} from "/scripts/core/signal.js";
import {StateManager} from "/scripts/core/state_manager.js";
import {Widget} from "/scripts/core/widget.js";

export class MultiStateToggle extends Widget {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._stateManager = new StateManager();
        this._stateManager.onValueChange.connect((value, data) => {
            this.innerText = data;

            this.onValueChange.fire(value);
        });

        this.tabIndex = 0;
        this.style.alignContent = "center";
        this.style.cursor = "pointer";
        this.style.height = "var(--widget-height)";
        this.style.maxWidth = "var(--widget-height)";
        this.style.minWidth = "var(--widget-height)";
        this.style.textAlign = "center";
        this.style.userSelect = "none";
        this.addEventListener("click", () => {
            this._stateManager.nextState();
        });
    }

    get states() {
        return this._stateManager.states;
    }

    get value() {
        return this._stateManager.value;
    }

    set states(value) {
        this._stateManager.states = value;
    }

    set value(value) {
        this._stateManager.value = value;
    }

    nextState() {
        this._stateManager.nextState();
    }
}
customElements.define("multi-state-toggle", MultiStateToggle);
