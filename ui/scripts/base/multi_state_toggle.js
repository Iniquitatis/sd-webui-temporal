import {Signal} from "/scripts/core/signal.js";
import {StateManager} from "/scripts/core/state_manager.js";
import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class MultiStateToggle extends Widget {
    static tag = "ce-multi-state-toggle";
    static css = `
        <self> {
            align-content: center;
            cursor: pointer;
            height: var(--widget-height);
            max-width: var(--widget-height);
            min-width: var(--widget-height);
            text-align: center;
            user-select: none;
        }
    `;

    constructor() {
        super();

        this.onValueChange = new Signal();

        this._stateManager = new StateManager();
        this._stateManager.onValueChange.connect((value, data) => {
            this.innerText = data;

            this.onValueChange.fire(value);
        });

        this.tabIndex = 0;
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
defineElement(MultiStateToggle);
