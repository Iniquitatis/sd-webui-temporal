* **Output directory** — general directory to which all of the extension's projects will be saved.
* **Project subdirectory** — current project's directory name inside of the output directory.
* **Frame count** — amount of cycles that will be performed, each of which producing a frame.
* **Save every N-th frame** — stride at which the frames will be saved.
* **Archive mode** — disable saving of metadata inside of each frame (such as prompt, seed, etc.) and enable maximum compression.
* **Start from scratch** — remove all rendered frames inside of the project directory, keeping the project settings intact.
* **Load session** — load settings and continue rendering starting from the last existing frame; does nothing if no frames were rendered in the current project.
* **Save session** — save current settings to the project directory.
    * **NOTE:** Currently saves all of the extension's frame preprocessing settings, some of the base settings (prompt, sampling method, img2img images, seed, etc.), and most of the ControlNet settings if it's installed.
* **Change rate** — amount of the new image that will be used to generate the next frame.
