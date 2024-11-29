function temporalGetCommunication(elem) {
    return parseInt(temporalGetClassValue(elem, "temporal-communication-"));
}

function temporalGetIndex(elem) {
    return parseInt(temporalGetClassValue(elem, "temporal-index-"));
}

function temporalQueryByIndex(index) {
    return gradioApp().querySelector(`.temporal-index-${index}`);
}

function temporalSend(communication, data) {
    let textArea = gradioApp().querySelector(`.temporal-communication-output.temporal-communication-${communication} textarea`);
    textArea.value = TEMPORAL_DEBUG ? JSON.stringify(data, null, 4) : JSON.stringify(data);

    let event = new Event("input", {bubbles: true});
    Object.defineProperty(event, "target", {value: textArea});
    textArea.dispatchEvent(event);
}

temporalInitialize(() => {
    gradioApp().querySelectorAll(".temporal-communication-input, .temporal-communication-output").forEach((box) => {
        if (TEMPORAL_DEBUG) box.classList.add("temporal-debug");
    });
});
