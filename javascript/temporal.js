const TEMPORAL_DEBUG = true;

function temporalGetClassValue(elem, prefix) {
    for (let cls of elem.classList) {
        if (cls.startsWith(prefix)) {
            return cls.slice(prefix.length);
        }
    }

    return null;
}

function temporalInitialize(callback) {
    document.addEventListener("DOMContentLoaded", () => {
        let observer = new MutationObserver(callback);
        observer.observe(gradioApp(), {childList: true, subtree: true});
    });
}
