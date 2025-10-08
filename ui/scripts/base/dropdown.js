import {Signal} from "/scripts/core/signal.js";
import {StateManager} from "/scripts/core/state_manager.js";
import {Widget} from "/scripts/core/widget.js";
import {clearElement, createElement, defineElement} from "/scripts/utils/dom.js";

export class Dropdown extends Widget {
    static tag = "ce-dropdown";
    static css = `
        <self> > select {
            width: 100%;
        }
    `;

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
        if (Array.isArray(value)) {
            let objectValue = {};

            for (let item of value) {
                objectValue[item] = item;
            }

            value = objectValue;
        }

        this._stateManager.states = value;
    }

    set value(value) {
        this._stateManager.value = value;
    }
}
defineElement(Dropdown);
