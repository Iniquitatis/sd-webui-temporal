**NOTE:** If the ControlNet extension is installed, its settings will also be saved alongside the project.  

* **Output** — common output parameters.
    * **Output directory** — main directory to which all of the extension's projects will be saved.
    * **Project subdirectory** — project's directory name inside the **Output directory**.
    * **Frame count** — amount of iterations that will be performed in the current session.
        * **NOTE:** This is _not_ a total amount of rendered frames. For example, if a project already has 57 frames rendered, then another `Frame count` frames will be rendered before stopping the process.
* **Project** — control over the session.
    * **Load parameters** — read parameters from the specified **Project subdirectory**, otherwise take those that are currently set in the UI.
    * **Continue from last frame** — continue rendering from the last existing frame inside the **Project subdirectory**, otherwise discard all of the previously rendered frames and start rendering from scratch.
    * **Autosave every N iterations** — save the project's data automatically over each N full iterations of the pipeline.
* **Live preview** — live preview options.
    * **Show only finalized frames** — set only the fully processed frames as live preview.
    * **Parallel index** — index of an image from the batch to preview.
        * **NOTE:** Can be set to `0` for previewing an entire batch.
