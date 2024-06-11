**NOTE:** If the ControlNet extension is installed, its settings will also be saved alongside the project.  

* **Project subdirectory** — project's directory name inside the **Output directory**.
* **Load parameters** — read parameters from the specified **Project subdirectory**, otherwise take those that are currently set in the UI.
* **Continue from last frame** — continue rendering from the last existing frame inside the **Project subdirectory**, otherwise discard all of the previously rendered frames and start rendering from scratch.
* **Iteration count** — amount of iterations that will be performed in the current session.
    * **NOTE:** This is _not_ a total amount of rendered frames. For example, if a project already has 57 frames rendered, then another `Iteration count` frames will be rendered before stopping the process.
