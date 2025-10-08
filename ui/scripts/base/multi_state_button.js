import {Button} from "/scripts/base/button.js";
import {Signal} from "/scripts/core/signal.js";
import {StateManager} from "/scripts/core/state_manager.js";
import {defineElement} from "/scripts/utils/dom.js";

export class MultiStateButton extends Button {
    static tag = "ce-multi-state-button";

    constructor() {
        super();

        this.onStateChange = new Signal();

        this._stateManager = new StateManager();
        this._stateManager.onValueChange.connect((value, data) => {
            this.label = data;

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
defineElement(MultiStateButton);
