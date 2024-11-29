// TODO: Investigate: If this function doesn't get called, then no scroll-jumping
// occurs at all, but accordions (not order list!) do not get resorted
function temporalUpdateReorderableList(index, data) {
    let list = temporalQueryByIndex(index);

    for (let accordionIndex of data) {
        list.appendChild(temporalQueryByIndex(accordionIndex));
    }
}

temporalInitialize(() => {
    let current = null;

    gradioApp().querySelectorAll(".temporal-reorderable-list").forEach((list) => {
        if (list.classList.contains("temporal-initialized")) return;

        list.querySelectorAll(".temporal-reorderable-accordion").forEach((accordion) => {
            let dragger = document.createElement("span");
            dragger.classList.add("temporal-reorderable-accordion-dragger");
            dragger.innerText = ":::";
            dragger.addEventListener("pointerdown", (event) => {
                event.stopPropagation();

                accordion.classList.add("temporal-dragged");

                current = accordion;
            });

            let checkbox = accordion.querySelector(".temporal-reorderable-accordion-checkbox");
            checkbox.addEventListener("click", (event) => {
                event.stopPropagation();
            });

            let labelWrap = accordion.querySelector(".label-wrap");
            labelWrap.insertBefore(dragger, labelWrap.lastChild);
            labelWrap.insertBefore(checkbox, labelWrap.lastChild);

            for (let specialCheckbox of accordion.querySelectorAll(".temporal-reorderable-accordion-special-checkbox")) {
                specialCheckbox.addEventListener("click", (event) => {
                    event.stopPropagation();
                });

                labelWrap.insertBefore(specialCheckbox, labelWrap.lastChild);
            }

            accordion.list = list;
        });
        list.classList.add("temporal-initialized");
    });

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

        temporalSend(
            temporalGetCommunication(current.list),
            [...current.list.querySelectorAll(".temporal-reorderable-accordion")].map(temporalGetIndex),
        );

        current.classList.remove("temporal-dragged");

        current = null;
    });
});
