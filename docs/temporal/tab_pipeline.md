**NOTE:** Almost all accordions within this tab can be reordered manually (using mouse or touchscreen, for example) to determine the order in which the modules will be invoked.  
**NOTE:** Order matters a lot. For example, invoking **Image filtering** before **Processing** will preprocess the frames, but otherwise it will postprocess them.  

* **Initial noise** — parameters of noise to use in a generation of the initial image if it's absent.
    * **Factor** — amount of noise that will be left unprocessed.
    * **Scale** — scale of the noise pattern; pixels.
    * **Octaves** — amount of progressively downscaled noise layers that will be mixed into a single one.
    * **Lacunarity** — downscale factor of each subsequent noise layer compared to a previous one.
    * **Persistence** — amplitude factor of each subsequent noise layer compared to a previous one.
* Generic options available to all modules:
    * **(✔️ to the left of a module name)** — determines if a module is enabled.
    * **Preview** — determines if module's results will be displayed in the UI.
* **Dampening** — blending of the current image towards the new image.
    * **Rate** — amount of blending between the current image and the new image (e.g. `0.5` will blend 50% of the new image into the current one).
* **Detailing** — an additional detailing pass that upscales the image and then scales it back, allowing for much higher precision at the cost of the processing speed.
    * **Scale** — upscaling factor.
        * **NOTE:** It doesn't affect the final output resolution, but rather the processing resolution itself.
    * **Sampling method** — same as the standard img2img option.
    * **Steps** — same as the standard img2img option.
    * **Denoising strength** — same as the standard img2img option.
* **Frame merging** — averaging of last generated frames.
    * **Buffer scale** — pixel scale of the internal buffer. Essentially, the precision at which the merging happens.
        * **NOTE:** It is uncertain whether it actually has a big impact on the result, but it certainly slows down the process a bit.
    * **Frame count** — amount of last generated frames to be blended together to produce a final frame.
        * **NOTE:** Slows down the morphing effect, increases the general middle-scale detail precision, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** filter).
    * **Trimming** — factor of distribution trimming (e.g. `0.25` trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
        * **NOTE:** The higher this value is, the longer it will take for any visible changes to occur: factor of `0.5` will require approximately half of **Frame count** first iterations.
    * **Easing** — frame averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each previous frame; at the value of `0` all frames will be blended evenly.
        * **NOTE:** This parameter won't have any effect if **Trimming** is greater than `0`.
    * **Preference** — "brightness preference" of the averaging algorithm. At minimum, it prefers the darkest colors, at maximum—the brightest ones.
        * **NOTE:** The greater the **Trimming** is, the less this parameter will affect the result.
* **Image filtering** — application of various filters to the image.
    * **NOTE:** All of the actual filter settings are located in the **Image Filtering** tab.
* **Limiting** — limiting of the difference between the previous and the current image.
    * **Mode** — limiting mode.
        * **clamp** — clamp the difference, cutting off anything higher than **Maximum difference**.
        * **compress** — compress the difference, "squashing" its range to **Maximum difference**.
    * **Maximum difference** — maximum difference between the values of the individual color channels.
        * **NOTE:** This value represents both positive and negative values.
* **Measuring** — measuring various image values, such as mean luma or standard deviation among every color channel.
    * **Plot every N-th frame** — stride at which the plots will be rendered.
    * **NOTE:** Resulting plots will be placed into the `<project subdirectory>/session/metrics` directory.
* **Processing** — the main Stable Diffusion processing procedure.
    * **NOTE:** Currently, all of the settings listed here are related to the averaging of several samples generated from a single frame.
    * **Sample count** — amount of samples to take for generating a frame.
        * **NOTE:** Reduces the jittering between the consecutive frames, increases the general middle-scale detail precision, multiplies amount of work to process each frame correspondingly, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** preprocessing effect).
    * **Batch size** — amount of samples to be calculated in parallel, potentially speeding up the process. If **Sample count** is not divisible by the batch size, it will be adjusted to the nearest divisible number (e.g. if **Sample count** is `9` and **Batch size** is `4`, then total sample count will equal to 12).
    * **Trimming** — factor of distribution trimming (e.g. `0.25` trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
    * **Easing** — sample averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each subsequent sample; at the value of `0` all samples will be blended evenly.
        * **NOTE:** This parameter won't have any effect if **Trimming** is greater than `0`.
    * **Preference** — "brightness preference" of the averaging algorithm. At minimum, it prefers the darkest colors, at maximum—the brightest ones.
        * **NOTE:** The greater the **Trimming** is, the less this parameter will affect the result.
* **Saving** — automatic saving of resulting images.
    * **Save every N-th frame** — stride at which the frames will be saved.
    * **Archive mode** — disable saving of metadata inside of each frame (such as prompt, seed, etc.) and enable maximum compression.
* **Video rendering** — automatic video rendering.
    * **Render draft/final every N-th frame** — stride at which a draft/final video will be rendered.
    * **Render draft/final on finish** — determines if a draft/final video will be rendered after finishing all iterations.
    * **NOTE:** All of the actual video configuration options are located in the **Video Rendering** tab.
