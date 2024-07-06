function getClassValue(elem, prefix) {
    for (let cls of elem.classList) {
        if (cls.startsWith(prefix)) {
            return cls.slice(prefix.length);
        }
    }

    return null;
}

function initializeReorderableLists() {
    let current = null;

    let observer = new MutationObserver(() => {
        gradioApp().querySelectorAll(".temporal-reorderable-list").forEach((list) => {
            if (list.classList.contains("temporal-initialized")) return;

            let listIndex = getClassValue(list, "temporal-index-");

            listTextbox = gradioApp().querySelector(`.temporal-reorderable-list-textbox.temporal-index-${listIndex}`);

            list.querySelectorAll(".temporal-reorderable-accordion").forEach((accordion) => {
                let accordionIndex = getClassValue(accordion, "temporal-index-");

                let dragger = document.createElement("span");
                dragger.classList.add("temporal-reorderable-accordion-dragger");
                dragger.classList.add(`temporal-index-${accordionIndex}`);
                dragger.innerText = ":::";
                dragger.addEventListener("pointerdown", (event) => {
                    event.stopPropagation();

                    accordion.classList.add("temporal-dragged");

                    current = accordion;
                });

                let checkbox = list.querySelector(`.temporal-reorderable-accordion-checkbox.temporal-index-${accordionIndex}`);
                checkbox.addEventListener("click", (event) => {
                    event.stopPropagation();
                });

                let labelWrap = accordion.querySelector(".label-wrap");
                labelWrap.insertBefore(checkbox.parentElement, labelWrap.firstChild);
                labelWrap.insertBefore(dragger, labelWrap.firstChild);

                let specialCheckbox = accordion.querySelector(".temporal-reorderable-accordion-special-checkbox");

                if (specialCheckbox != null) {
                    specialCheckbox.addEventListener("click", (event) => {
                        event.stopPropagation();
                    });

                    labelWrap.insertBefore(specialCheckbox, labelWrap.lastChild);
                }

                accordion.checkbox = checkbox;
                accordion.list = list;
                accordion.listTextbox = listTextbox;
            });
            list.classList.add("temporal-initialized");
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

        parent.querySelectorAll(".temporal-reorderable-accordion").forEach((other) => {
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

        let textArea = current.listTextbox.querySelector("textarea");
        textArea.value =
            [...current.list.querySelectorAll(".temporal-reorderable-accordion")]
            .map((x) => getClassValue(x, "temporal-index-"))
            .join("|");

        let inputEvent = new Event("input", {bubbles: true});
        Object.defineProperty(inputEvent, "target", {value: textArea});
        textArea.dispatchEvent(inputEvent);

        current.classList.remove("temporal-dragged");

        current = null;
    });
}

function updateReorderableListOrder(listIndex, accordionIndexString) {
    let list = gradioApp().querySelector(`.temporal-reorderable-list.temporal-index-${listIndex}`);

    for (let index of accordionIndexString.split("|")) {
        list.appendChild(list.querySelector(`.temporal-reorderable-accordion.temporal-index-${index}`));
    }
}

document.addEventListener("DOMContentLoaded", () => {
    initializeReorderableLists();
});
