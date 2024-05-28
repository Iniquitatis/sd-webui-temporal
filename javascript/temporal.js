function getClassValue(elem, prefix) {
    for (let cls of elem.classList) {
        if (cls.startsWith(prefix)) {
            return cls.slice(prefix.length);
        }
    }

    return null;
}

function initializeModuleLists() {
    let current = null;

    let observer = new MutationObserver(() => {
        gradioApp().querySelectorAll(".temporal-module-list").forEach((moduleList) => {
            if (moduleList.classList.contains("temporal-initialized")) return;

            let moduleListIndex = getClassValue(moduleList, "temporal-index-");

            moduleListDropdown = gradioApp().querySelector(`.temporal-module-list-dropdown.temporal-index-${moduleListIndex}`);
            moduleListTextbox = gradioApp().querySelector(`.temporal-module-list-textbox.temporal-index-${moduleListIndex}`);

            moduleList.querySelectorAll(".temporal-module-accordion").forEach((accordion) => {
                let accordionIndex = getClassValue(accordion, "temporal-index-");

                let dragger = document.createElement("span");
                dragger.classList.add("temporal-module-accordion-dragger");
                dragger.classList.add(`temporal-index-${accordionIndex}`);
                dragger.innerText = ":::";
                dragger.addEventListener("pointerdown", (event) => {
                    event.stopPropagation();

                    accordion.classList.add("temporal-dragged");

                    current = accordion;
                });

                let checkbox = gradioApp().querySelector(`.temporal-module-accordion-checkbox.temporal-index-${accordionIndex}`);
                checkbox.addEventListener("click", (event) => {
                    event.stopPropagation();
                });

                let labelWrap = accordion.querySelector(".label-wrap");
                labelWrap.insertBefore(checkbox.parentElement, labelWrap.firstChild);
                labelWrap.insertBefore(dragger, labelWrap.firstChild);

                accordion.checkbox = checkbox;
                accordion.moduleList = moduleList;
                accordion.moduleListDropdown = moduleListDropdown;
                accordion.moduleListTextbox = moduleListTextbox;
            });
            moduleList.classList.add("temporal-initialized");
        });
    });
    observer.observe(gradioApp(), {childList: true, subtree: true});

    window.addEventListener("touchmove", (event) => {
        if (!current) return;

        event.stopPropagation();
        event.preventDefault();
    }, {passive: false});
    window.addEventListener("pointermove", (event) => {
        if (!current) return;

        event.stopPropagation();

        let parent = current.parentElement;

        parent.querySelectorAll(".temporal-module-accordion").forEach((other) => {
            if (current == other) return;

            let selfRect = current.getBoundingClientRect();
            let otherRect = other.getBoundingClientRect();

            if (selfRect.top < otherRect.top && event.clientY > otherRect.top) {
                parent.insertBefore(other, current);
            }

            if (selfRect.top > otherRect.top && event.clientY < otherRect.bottom) {
                parent.insertBefore(current, other);
            }
        })
    });
    window.addEventListener("pointerup", (event) => {
        if (!current) return;

        event.stopPropagation();

        let textArea = current.moduleListTextbox.querySelector("textarea");
        textArea.value =
            [...current.moduleList.querySelectorAll(".temporal-module-accordion")]
            .map((x) => getClassValue(x, "temporal-key-"))
            .join("|");

        let inputEvent = new Event("input", {bubbles: true});
        Object.defineProperty(inputEvent, "target", {value: textArea});
        textArea.dispatchEvent(inputEvent);

        current.classList.remove("temporal-dragged");

        current = null;
    });
}

function updateModuleListOrder(index, keys) {
    let moduleList = gradioApp().querySelector(`.temporal-module-list.temporal-index-${index}`);

    for (let key of keys) {
        moduleList.appendChild(moduleList.querySelector(`.temporal-module-accordion.temporal-key-${key}`));
    }
}

document.addEventListener("DOMContentLoaded", () => {
    initializeModuleLists();
});
