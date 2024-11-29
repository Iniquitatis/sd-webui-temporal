function safeMax(array, fallback, predicate = null) {
    let max = Math.max(...(predicate ? array.map(predicate) : array));
    return max != -Infinity ? max : fallback;
}

//==============================================================================

class TemporalSignal {
    constructor() {
        this._listeners = [];
    }

    connect(func) {
        this._listeners.push(func);
        return func;
    }

    disconnect(func) {
        this._listeners.splice(this._listeners.indexOf(func), 1);
    }

    fire(...args) {
        for (let listener of this._listeners) {
            listener(...args);
        }
    }
}

//==============================================================================

class TemporalMultiStateToggle extends HTMLElement {
    constructor(states) {
        super();

        // NOTE: Yes, `states` is a `value:caption` dictionary
        this._values = Object.keys(states);
        this._captions = states;
        this._index = 0;

        this.addEventListener("click", () => {
            this._index++;
            this._index %= this._values.length;
            this._refresh();

            this.onValueChange.fire(this.value);
        });

        this.onValueChange = new TemporalSignal();

        this._refresh();
    }

    get value() {
        return this._values[this._index];
    }

    set value(value) {
        this._index = this._values.indexOf(value);
        this._refresh();

        this.onValueChange.fire(this.value);
    }

    _refresh() {
        this.innerText = this._captions[this.value];
    }
}
customElements.define("temporal-multi-state-toggle", TemporalMultiStateToggle);

//==============================================================================

class TemporalValueInput extends HTMLElement {
    static _globalId = 0;

    constructor() {
        super();

        this._id = TemporalValueInput._globalId;
        TemporalValueInput._globalId++;

        this.onValueChange = new TemporalSignal();
    }

    get defaultValue() {
        switch (this._type) {
            case "bool":  return false;
            case "int":
            case "float": return 0;
            case "text":  return "";
            case "color": return "#000000";
            case "enum":  return "";
            case "list":  return "";
        }
    }

    get value() {
        switch (this._type) {
            case "bool":   return this._input.checked;
            case "int":
            case "float":  return this._input.valueAsNumber;
            case "text":   return this._input.value;
            case "color":  return this._input.value;
            case "enum":   return this._input.options[this._input.selectedIndex].text;
            case "list":   return this._input.value;
        }
    }

    set choices(values) {
        switch (this._type) {
            case "enum": {
                for (let value in values) {
                    let caption = values[value];
                    let option = document.createElement("option");
                    option.value = value;
                    option.innerText = caption;
                    this._input.appendChild(option);
                }
            } break;

            case "list": {
                for (let value of values) {
                    let option = document.createElement("option");
                    option.value = value;
                    option.innerText = value;
                    this._data.appendChild(option);
                }
            } break;
        }
    }

    set enabled(value) {
        this._input.disabled = !value;
    }

    set maximum(value) {
        this._input.max = value;
    }

    set minimum(value) {
        this._input.min = value;
    }

    set step(value) {
        this._input.step = value;
    }

    set type(value) {
        this._type = value;

        while (this.childElementCount > 0) this.removeChild(this.lastChild);

        switch (value) {
            case "bool": {
                this._input = document.createElement("input");
                this._input.type = "checkbox";
                this._input.checked = this.defaultValue;
                this._input.addEventListener("change", () => this.onValueChange.fire(this.value));
                this.appendChild(this._input);
            } break;

            case "int": {
                this._input = document.createElement("input");
                this._input.type = "number";
                this._input.step = 1;
                this._input.valueAsNumber = this.defaultValue;
                this._input.addEventListener("change", () => this.onValueChange.fire(this.value));
                this.appendChild(this._input);
            } break;

            case "float": {
                this._input = document.createElement("input");
                this._input.type = "number";
                this._input.step = 0.01;
                this._input.valueAsNumber = this.defaultValue;
                this._input.addEventListener("change", () => this.onValueChange.fire(this.value));
                this.appendChild(this._input);
            } break;

            case "text": {
                this._input = document.createElement("input");
                this._input.type = "text";
                this._input.value = this.defaultValue;
                this._input.addEventListener("change", () => this.onValueChange.fire(this.value));
                this.appendChild(this._input);
            } break;

            case "color": {
                this._input = document.createElement("input");
                this._input.type = "color";
                this._input.value = this.defaultValue;
                this._input.addEventListener("change", () => this.onValueChange.fire(this.value));
                this.appendChild(this._input);
            } break;

            case "enum": {
                this._input = document.createElement("select");
                this._input.value = this.defaultValue;
                this._input.addEventListener("change", () => this.onValueChange.fire(this.value));
                this.appendChild(this._input);
            } break;

            case "list": {
                this._data = document.createElement("datalist");
                this._data.id = `temporal-data-${this._id}`;
                this.appendChild(this._data);

                this._input = document.createElement("input");
                this._input.type = "text";
                this._input.setAttribute("list", this._data.id);
                this._input.value = this.defaultValue;
                this._input.addEventListener("change", () => this.onValueChange.fire(this.value));
                this.appendChild(this._input);
            } break;
        }
    }

    set value(value) {
        switch (this._type) {
            case "bool":  this._input.checked = value; break;
            case "int":
            case "float": this._input.valueAsNumber = value; break;
            case "text":  this._input.value = value; break;
            case "color": this._input.value = value; break;
            case "enum":  this._input.value = [...this._input.options].findIndex(
                (option) => option.text == value
            ); break;
            case "list":  this._input.value = value; break;
        }
    }
}
customElements.define("temporal-value-input", TemporalValueInput);

//==============================================================================

class TemporalAnimationEditor extends HTMLElement {
    constructor() {
        super();

        this._properties = {};

        this._timeline = new TemporalAnimationEditorTimeline();
        this._timeline.onCellClick.connect((frame) => {
            this._timeline.highlightCell(frame - 1);

            for (let track of this.tracks) {
                track.currentFrame = frame;
                track.highlightCell(frame - 1);
            }
        });
        this.appendChild(this._timeline);

        this._addTrackBtn = document.createElement("button");
        this._addTrackBtn.innerText = "+";
        this._addTrackBtn.addEventListener("click", () => this.addTrack());
        this.appendChild(this._addTrackBtn);

        this._removeTrackBtn = document.createElement("button");
        this._removeTrackBtn.innerText = "-";
        this._removeTrackBtn.addEventListener("click", () => this.removeChild(this.tracks[this.tracks.length - 1]));
        this.appendChild(this._removeTrackBtn);

        this.onChange = new TemporalSignal();
    }

    get tracks() {
        return [...this.querySelectorAll("temporal-anim-track")];
    }

    addTrack(data = null) {
        let result = new TemporalAnimationEditorTrack(this._properties);
        if (data) result.readData(data);
        result.onLengthChange.connect(() => {
            this._timeline.length = safeMax(this.tracks, 0, (track) => track.length);
        });
        result.onChange.connect(() => {
            this.onChange.fire(this);
        });
        this.insertBefore(result, this.lastChild);

        this.onChange.fire(this);

        return result;
    }

    readData(data) {
        this._properties = data.properties;
        this._timeline.length = safeMax(data.tracks, 0, (track) =>
            safeMax(track.keyframes, 0, (keyframe) => keyframe.frame)
        );

        data.tracks.forEach(this.addTrack);

        this.onChange.fire(this);
    }

    storeData() {
        return {
            tracks: this.tracks.map((track) => track.storeData()),
        }
    }
}
customElements.define("temporal-anim-editor", TemporalAnimationEditor);

//==============================================================================

class TemporalAnimationEditorRow extends HTMLElement {
    constructor() {
        super();

        this._highlightedIndex = -1;

        this.header = document.createElement("div");
        this.header.className = "temporal-anim-row-header";
        this.appendChild(this.header);

        this.data = document.createElement("div");
        this.data.className = "temporal-anim-row-data";
        this.appendChild(this.data);

        this.onLengthChange = new TemporalSignal();
        this.onCellClick = new TemporalSignal();
    }

    get cells() {
        return this.data.children;
    }

    get length() {
        return this.data.childElementCount;
    }

    set length(value) {
        while (this.data.childElementCount > value) this.removeCell();
        while (this.data.childElementCount < value) this.addCell();
    }

    addCell() {
        let result = document.createElement("div");
        result.className = "temporal-anim-row-cell";
        result.addEventListener("click", (event) => {
            this.onCellClick.fire(parseInt(event.target.innerHTML));
        });
        this.data.appendChild(result);

        if (this._highlightedIndex == this.length - 1) {
            result.classList.add("highlighted");
        }

        this.onLengthChange.fire(this.length);

        return result;
    }

    highlightCell(index) {
        this._highlightedIndex = index;

        for (let cell of this.cells) {
            cell.classList.remove("highlighted");
        }

        if (index >= this.length) return;

        let cell = this.cells[index];
        cell.classList.add("highlighted");
    }

    removeCell() {
        if (this.data.childElementCount == 0) return;

        this.data.removeChild(this.data.lastChild);

        this.onLengthChange.fire(this.length);
    }
}
customElements.define("temporal-anim-row", TemporalAnimationEditorRow);

//==============================================================================

class TemporalAnimationEditorTimeline extends TemporalAnimationEditorRow {
    constructor() {
        super();

        this.header.innerText = "Animation";
    }

    addCell() {
        let result = super.addCell();
        result.innerHTML = `${this.data.childElementCount}`;

        return result;
    }
}
customElements.define("temporal-anim-timeline", TemporalAnimationEditorTimeline);

//==============================================================================

class TemporalAnimationEditorTrack extends TemporalAnimationEditorRow {
    constructor(properties) {
        super();

        this._propertyTypes = properties;
        this._currentFrame = 0;

        this._property = new TemporalValueInput();
        this._property.type = "list";
        // FIXME: Temporary randomness
        // this._property.choices = Math.random() < 0.5 ? Object.keys(properties) : ["A", "b", "C"];
        this._property.onValueChange.connect((value) => {
            this._value.type = this._propertyTypes[value];
            this._refreshValue();

            this.onChange.fire(this);
        });
        this.header.appendChild(this._property);

        this._length = new TemporalValueInput();
        this._length.type = "int";
        this._length.minimum = 0;
        this._length.maximum = 100;
        this._length.step = 1;
        this._length.onValueChange.connect((value) => {
            this.length = value;

            this.onChange.fire(this);
        });
        this.header.appendChild(this._length);

        this._interpolation = new TemporalMultiStateToggle({
            "linear": "\u27cb",
            "smoothstep": "\u2937",
            "smootherstep": "\u293f",
            "step": "\u{1328d}",
            "step_start": "\u2a3d",
            "step_end": "\u2a3c",
        });
        this._interpolation.onValueChange.connect(() => {
            this.onChange.fire(this);
        });
        this.header.appendChild(this._interpolation);

        this._bounds = new TemporalMultiStateToggle({
            "clamp": "\u22a3",
            "repeat": "\u27f3",
            "mirror": "\u27db",
        });
        this._bounds.onValueChange.connect((value) => {
            this.onChange.fire(this);
        });
        this.header.appendChild(this._bounds);

        this._value = new TemporalValueInput();
        this._value.type = "text";
        this._value.onValueChange.connect((value) => {
            this.keyframes[this._currentFrame].value = value;

            this.onChange.fire(this);
        });
        this.header.appendChild(this._value);

        this.onLengthChange.connect((length) => {
            this._length.value = length;
            this._refreshValue();

            this.onChange.fire(this);
        });

        this.onChange = new TemporalSignal();
    }

    get keyframes() {
        return [...this.data.querySelectorAll("temporal-anim-keyframe")];
    }

    set currentFrame(value) {
        this._currentFrame = value;
        this._refreshValue();
    }

    addCell() {
        let result = super.addCell();

        let keyframe = new TemporalAnimationEditorKeyframe();
        keyframe.frame = this.length;
        keyframe.value = this._value.defaultValue;
        keyframe.onStateChange.connect(() => {
            this.onChange.fire(this);
        });
        result.appendChild(keyframe);

        return result;
    }

    readData(data) {
        this._property.value = data.name;
        this._interpolation.value = data.interpolation;
        this._bounds.value = data.bounds;
        this._value.type = this._propertyTypes[data.name] || "text";
        this._refreshValue();

        this.length = safeMax(data.keyframes, 0, (keyframe) => keyframe.frame);

        data.keyframes.forEach((data) => {
            this.keyframes[data.frame - 1].readData(data);
        });
    }

    storeData() {
        return {
            name: this._property.value,
            interpolation: this._interpolation.value,
            bounds: this._bounds.value,
            keyframes: this.keyframes
                .filter((keyframe) => keyframe.enabled)
                .map((keyframe) => keyframe.storeData()),
        };
    }

    _refreshValue() {
        if (this._currentFrame == 0) return;

        let keyframe = this.keyframes[this._currentFrame];

        this._value.value = keyframe.value;
        this._value.enabled = keyframe.enabled;
    }
}
customElements.define("temporal-anim-track", TemporalAnimationEditorTrack);

//==============================================================================

class TemporalAnimationEditorKeyframe extends HTMLElement {
    constructor() {
        super();

        this._state = new TemporalMultiStateToggle({"unset": "", "set": "\u25c6"});
        this._state.value = "unset";
        this._state.onValueChange.connect((value) => this.onStateChange.fire(value == "set"));
        this.appendChild(this._state);

        this.frame = -1;
        this.value = null;

        this.onStateChange = new TemporalSignal();
    }

    get enabled() {
        return this._state.value == "set";
    }

    readData(data) {
        this._state.value = "set";

        this.frame = data.frame;
        this.value = data.value;
    }

    storeData() {
        return {
            frame: this.frame,
            value: this.value,
        }
    }
}
customElements.define("temporal-anim-keyframe", TemporalAnimationEditorKeyframe);

//==============================================================================

// let animDef = {
//     "properties": {
//         "parameters.strength": "number",
//         "pipeline.modules[0].enabled": "bool",
//         "pipeline.modules[0].amount": "number",
//         "pipeline.modules[1].enabled": "bool",
//         "pipeline.modules[1].amount": "number",
//         "pipeline.modules[2].enabled": "bool",
//         "pipeline.modules[2].amount": "number",
//         "pipeline.modules[2].color": "color",
//     },
//     "tracks": [
//         {
//             "name": "parameters.strength",
//             "interpolation": "linear",
//             "bounds": "repeat",
//             "keyframes": [
//                 {"frame": 1, "value": 1.0},
//                 {"frame": 3, "value": 0.9},
//                 {"frame": 6, "value": 0.6},
//                 {"frame": 8, "value": 0.5},
//             ],
//         },
//         {
//             "name": "pipeline.modules[0].enabled",
//             "interpolation": "step_start",
//             "bounds": "repeat",
//             "keyframes": [
//                 {"frame": 1, "value": true},
//                 {"frame": 7, "value": false},
//             ],
//         },
//         {
//             "name": "pipeline.modules[2].amount",
//             "interpolation": "smoothstep",
//             "bounds": "mirror",
//             "keyframes": [
//                 {"frame": 1, "value": 0.75},
//                 {"frame": 2, "value": 0.5},
//                 {"frame": 6, "value": 0.0},
//             ],
//         },
//         {
//             "name": "pipeline.modules[2].color",
//             "interpolation": "linear",
//             "bounds": "clamp",
//             "keyframes": [],
//         },
//     ],
// };

// TODO: In-out communication boxes?
// function temporalUpdateAnimationEditor(index, data) {
//     let editor = temporalQueryByIndex(index);
//     editor.readData(data);
// }

temporalInitialize(() => {
    gradioApp().querySelectorAll("temporal-anim-editor").forEach((editor) => {
        if (editor.classList.contains("temporal-initialized")) return;

        editor.onChange.connect(() => {
            temporalSend(temporalGetCommunication(editor), editor.storeData());
        });

        editor.classList.add("temporal-initialized");
    });
});

// document.addEventListener("DOMContentLoaded", () => {
//     let output = document.createElement("code");
//     output.style.color = "light-dark(black, white)";
//     output.style.whiteSpaceCollapse = "preserve";

//     let editor = new TemporalAnimationEditor();
//     //editor.animation = animDef;
//     editor.onChange.connect((value) => output.innerText = JSON.stringify(value, null, 4));
//     document.body.appendChild(editor);

//     document.body.appendChild(output);
// });
