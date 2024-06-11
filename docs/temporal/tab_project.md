**NOTE:** If the ControlNet extension is installed, its settings will also be saved alongside the project.  

* **Project** — current project. Name can be entered manually to create a new one after the processing begins.
    * 🔄 — update project list.
    * 📂 — load UI parameters from the currently selected project.
    * 🗑️ — delete currently selected project.
* **Session** — current session parameters.
    * **Load parameters** — read parameters from the specified **Project subdirectory**, otherwise take those that are currently set in the UI.
    * **Continue from last frame** — continue rendering from the last existing frame inside the **Project subdirectory**, otherwise discard all of the previously rendered frames and start rendering from scratch.
    * **Iteration count** — amount of iterations that will be performed in the current session.
        * **NOTE:** This is _not_ a total amount of rendered frames. For example, if a project already has 57 frames rendered, then another `Iteration count` frames will be rendered before stopping the process.
* **Information** — information about the currently selected project.
    * **Description** — textual description of the currently selected project.
    * **Last frames** — small gallery showing the last rendered frames from the currently selected project.
    * **Parallel index** — index of an image set to preview in the gallery.
* **Tools** — various utilities for managing the currently selected project.
    * **New name** — new name for the currently selected project.
        * ✔️ — rename currently selected project.
        * **NOTE:** This will rename both the project's folder and the rendered videos.
    * **Delete intermediate frames** — delete all frames in the project's folder other than the first one and the last one.
    * **Delete session data** — delete the temporary session data.
        * **WARNING:** This will make the project unable to be continued—only restarting with the same parameters will be possible.
