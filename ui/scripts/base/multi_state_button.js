import {Button} from "/scripts/base/button.js";
import {Signal} from "/scripts/core/signal.js";
import {StateManager} from "/scripts/core/state_manager.js";

export class MultiStateButton extends Button {
    constructor() {
        super();

        this.onStateChange = new Signal();

        this._stateManager = new StateManager();
        this._stateManager.onValueChange.connect((value, data) => {
            this._button.innerText = data;

            this.onStateChange.fire(value);
        });

        this.onClick.connect(() => {
            this._stateManager.nextState();
        });
    }

    get state() {
        return this._stateManager.value;
    }

    get states() {
        return this._stateManager.states;
    }

    set state(value) {
        this._stateManager.value = value;
    }

    set states(value) {
        this._stateManager.states = value;
    }

    nextState() {
        this._stateManager.nextState();
    }
}
customElements.define("ce-multi-state-button", MultiStateButton);
