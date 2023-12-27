**NOTE:** If ControlNet extension is installed, its settings will also be saved alongside the project.

* **Output** — common output parameters.
    * **Output directory** — general directory to which all of the extension's projects will be saved.
    * **Project subdirectory** — current project's directory name inside of the output directory.
    * **Frame count** — amount of cycles that will be performed, each of which producing a frame.
    * **Save every N-th frame** — stride at which the frames will be saved.
    * **Archive mode** — disable saving of metadata inside of each frame (such as prompt, seed, etc.) and enable maximum compression.
* **Processing** — processing parameters.
    * **Noise for first frame** — use noise as a first frame instead of a newly generated image if the initial img2img image is absent.
    * **Use Stable Diffusion** — process frames with Stable Diffusion after preprocessing them, otherwise simply output preprocessed frames.
        * **NOTE:** Disabling this option might be useful for tweaking the preprocessing parameters and checking how they affect the results. Most of the time it should be enabled, though, unless one wants to use this extension as a sort of a simple image animator.
* **Multisampling** — averaging of several samples generated from a single frame.
    * **Sample count** — amount of samples to take for generating a frame.
        * **NOTE:** Reduces the jittering between the consecutive frames, increases the general middle-scale detail precision, multiplies amount of work to process each frame correspondingly, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** preprocessing effect).
        * **Batch size** — amount of samples to be calculated in parallel, potentially speeding up the process. If **Sample count** is not divisible by the batch size, it will be adjusted to the nearest divisible number (e.g. if **Sample count** is 9 and **Batch size** is 4, then total sample count will equal to 12).
    * **Algorithm** — averaging algorithm.
        * **harmonic_mean** — mean that prefers the darkest colors.
        * **geometric_mean** — mean that prefers the darker colors.
        * **arithmetic_mean** — standard mean.
        * **root_mean_square** — mean that prefers the brightest colors.
        * **median** — produces sharp results, but makes images "oily" and "smudgy", leaving out fine details and abrupt composition changes.
    * **Easing** — sample averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each subsequent sample; at the value of 0 all samples will be blended evenly.
        * **NOTE:** This parameter is relevant only for mean algorithms.
    * **Trimming** — factor of distribution trimming (e.g. 0.25 trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
        * **NOTE:** As of now, the **Easing** parameter will not work if this factor is greater than 0.
* **Frame merging** — averaging of several last generated frames.
    * **Frame count** — amount of last generated frames to be blended together to produce a final frame.
        * **NOTE:** Slows down the morphing effect, increases the general middle-scale detail precision, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** preprocessing effect).
    * **Algorithm** — averaging algorithm.
        * **harmonic_mean** — mean that prefers the darkest colors.
        * **geometric_mean** — mean that prefers the darker colors.
        * **arithmetic_mean** — standard mean.
        * **root_mean_square** — mean that prefers the brightest colors.
        * **median** — produces sharp results, but makes images "oily" and "smudgy", leaving out fine details and abrupt composition changes.
            * **NOTE:** As of now, the results won't be visible for approximately half of **Frame count** first iterations.
    * **Easing** — frame averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each previous frame; at the value of 0 all frames will be blended evenly.
        * **NOTE:** This parameter is relevant only for mean algorithms.
    * **Trimming** — factor of distribution trimming (e.g. 0.25 trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
        * **NOTE:** As of now, the **Easing** parameter will not work if this factor is greater than 0.
* **Project** — control over the session.
    * **Load parameters** — read parameters from the specified project directory, otherwise take those that are currently set in the UI.
    * **Continue from last frame** — continue rendering from the last rendered frame, otherwise remove all previously rendered frames and start rendering from scratch.
