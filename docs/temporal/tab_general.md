**NOTE:** If ControlNet extension is installed, its settings will also be saved alongside the project.

* **Output** — common output parameters.
    * **Output directory** — general directory to which all of the extension's projects will be saved.
    * **Project subdirectory** — current project's directory name inside of the output directory.
    * **Frame count** — amount of cycles that will be performed, each of which producing a frame.
    * **Save every N-th frame** — stride at which the frames will be saved.
    * **Archive mode** — disable saving of metadata inside of each frame (such as prompt, seed, etc.) and enable maximum compression.
* **Initial noise** — parameters of noise to use in a generation of the initial image if it's absent.
    * **Factor** — amount of noise that will be left unprocessed.
    * **Scale** — scale of the noise pattern; pixels.
    * **Octaves** — amount of progressively downscaled noise layers that will be mixed into a single one.
    * **Persistence** — downscale factor of each subsequent noise layer compared to a previous one.
    * **Lacunarity** — amplitude factor of each subsequent noise layer compared to a previous one.
* **Processing** — processing parameters.
    * **Use Stable Diffusion** — process frames with Stable Diffusion after preprocessing them, otherwise simply output preprocessed frames.
        * **NOTE:** Disabling this option might be useful for tweaking the preprocessing parameters and checking how they affect the results. Most of the time it should be enabled, though, unless one wants to use this extension as a sort of a simple image animator.
* **Multisampling** — averaging of several samples generated from a single frame.
    * **Sample count** — amount of samples to take for generating a frame.
        * **NOTE:** Reduces the jittering between the consecutive frames, increases the general middle-scale detail precision, multiplies amount of work to process each frame correspondingly, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** preprocessing effect).
        * **Batch size** — amount of samples to be calculated in parallel, potentially speeding up the process. If **Sample count** is not divisible by the batch size, it will be adjusted to the nearest divisible number (e.g. if **Sample count** is 9 and **Batch size** is 4, then total sample count will equal to 12).
    * **Trimming** — factor of distribution trimming (e.g. 0.25 trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
    * **Easing** — sample averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each subsequent sample; at the value of 0 all samples will be blended evenly.
        * **NOTE:** This parameter won't have any effect if the **Trimming** parameter is greater than 0.
    * **Preference** — "brightness preference" of the averaging algorithm. At minimum, it prefers the darkest colors, at maximum—the brightest ones.
        * **NOTE:** The greater the **Trimming** parameter is, the less this parameter will affect the result.
* **Detailing** — an additional detailing pass that upscales the image and then scales it back, allowing for much higher precision at the cost of the processing speed.
    * **Enabled** — determines whether the detailing is enabled or not.
    * **Scale** — upscaling factor.
        * **NOTE:** It doesn't affect the final output resolution, but rather the processing resolution itself.
    * **Scale buffer** — determines whether the **Frame merging** happens at a higher precision.
        * **NOTE:** It is uncertain whether it actually has a big impact on the result, but it certainly slows down the process a bit.
    * **Sampling method** — same as the usual img2img option.
    * **Steps** — same as the usual img2img option.
    * **Denoising strength** — same as the usual img2img option.
* **Frame merging** — averaging of several last generated frames.
    * **Frame count** — amount of last generated frames to be blended together to produce a final frame.
        * **NOTE:** Slows down the morphing effect, increases the general middle-scale detail precision, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** preprocessing effect).
    * **Trimming** — factor of distribution trimming (e.g. 0.25 trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
        * **NOTE:** The higher this value is, the longer it will take for any visible changes to occur: factor of 0.5 will require approximately half of **Frame count** first iterations.
    * **Easing** — frame averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each previous frame; at the value of 0 all frames will be blended evenly.
        * **NOTE:** This parameter won't have any effect if the **Trimming** parameter is greater than 0.
    * **Preference** — "brightness preference" of the averaging algorithm. At minimum, it prefers the darkest colors, at maximum—the brightest ones.
        * **NOTE:** The greater the **Trimming** parameter is, the less this parameter will affect the result.
* **Project** — control over the session.
    * **Load parameters** — read parameters from the specified project directory, otherwise take those that are currently set in the UI.
    * **Continue from last frame** — continue rendering from the last rendered frame, otherwise remove all previously rendered frames and start rendering from scratch.
