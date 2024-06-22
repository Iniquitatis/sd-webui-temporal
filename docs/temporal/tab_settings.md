* **Output** — output options.
    * **Output directory** — main directory to which all of the extension's projects will be saved.
    * **Autosave every N iterations** — save the project's data automatically over each N full iterations of the pipeline.
* **Live preview** — live preview options.
    * **Show only finished images** — set only the fully processed frames as live preview.
    * **Parallel index** — index of an image from the batch to preview.
        * **NOTE:** Can be set to `0` for previewing an entire batch.
* **Processing** — processing options.
    * **Pixels per batch** — upper limit of pixels (`width * height`) below which images will be rendered in parallel, potentially speeding up the processing. For example, if this option is set to `1048576` (equals to an image of size `1024x1024`, `2048x512`, and so on), then exactly four images of size `512x512` can be processed in a single batch.
* **UI** — user interface options.
    * **Preset sorting order** — determines how presets are sorted inside the preset menu.
    * **Project sorting order** — determines how projects are sorted inside the project menu.
    * **Gallery size** — amount of images to show in the project preview gallery.
