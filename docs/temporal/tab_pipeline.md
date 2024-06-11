**NOTE:** Almost all accordions within this tab can be reordered manually (using mouse or touchscreen, for example) to determine the order in which the modules will be invoked.  
**NOTE:** Order matters a lot. For example, applying **Noise overlay** before **Color correction** will make the noise color corrected, but otherwise the noise will be applied _on top_ of a color corrected image.  

All modules in this tab are marked with the following icons:
* ✨ — image filter that affects the image appearance directly.
* 🕓 — temporal module that takes multiple subsequent rendered frames into account in order to work. May take several iterations for the effect to be visible at all.
* 🛠 — tool module that doesn't directly affect an image, but rather does some action such as saving an image.
* 🧬 — neural network module that invokes Stable Diffusion in order to process an image.

* **Initial noise** — parameters of noise to use in a generation of the initial image if it's absent.
    * **Factor** — amount of noise that will be left unprocessed.
    * **Scale** — scale of the noise pattern; pixels.
    * **Octaves** — amount of progressively downscaled noise layers that will be mixed into a single one.
    * **Lacunarity** — downscale factor of each subsequent noise layer compared to a previous one.
    * **Persistence** — amplitude factor of each subsequent noise layer compared to a previous one.
* **Parallel** — amount of images to create/process in parallel.
* Generic options available to all modules:
    * **(✔️ to the left of a module name)** — determines if a module is enabled.
    * **Preview** — determines if module's results will be displayed in the UI.
* Generic options available to all filter modules:
    * **Amount** — amount of the processed image to be mixed in into the frame.
        * **Relative** — determines if the amount will be multiplied by the **img2img** denoising strength.
    * **Blend mode** — blending mode of the processed image.
    * **Mask** — an image that determines which areas will be processed by a filter; black—unprocessed, white—fully processed.
        * **Normalized** — normalization of the mask image to use the full range from black to white.
        * **Inverted** — inversion of the mask image.
        * **Blurring** — blurring radius of the mask image.
* **Averaging** — averaging of last generated frames.
    * **Frame count** — amount of last generated frames to be blended together to produce a final frame.
        * **NOTE:** Slows down the morphing effect, increases the general middle-scale detail precision, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** filter).
    * **Trimming** — factor of distribution trimming (e.g. `0.25` trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
        * **NOTE:** The higher this value is, the longer it will take for any visible changes to occur: factor of `0.5` will require approximately half of **Frame count** first iterations.
    * **Easing** — frame averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each previous frame; at the value of `0` all frames will be blended evenly.
        * **NOTE:** This parameter won't have any effect if **Trimming** is greater than `0`.
    * **Preference** — "brightness preference" of the averaging algorithm. At minimum, it prefers the darkest colors, at maximum—the brightest ones.
        * **NOTE:** The greater the **Trimming** is, the less this parameter will affect the result.
* **Blurring** — frame blurring.
    * **Radius** — blurring radius.
* **Color balancing** — common color balancing.
    * **Brightness** — target brightness of the frame.
    * **Contrast** — target contrast of the frame.
    * **Saturation** — target saturation of the frame.
* **Color correction** — various parameters to keep frame's palette normalized.
    * **Reference image** — an image that will be used to match histograms. Simply put, an overall color balance of the frame will be matched against this image.
    * **Normalize contrast** — normalize the contrast curve of the frame so that it's in range of 0.0–1.0.
    * **Equalize histogram** — equalize the image histogram, distributing the color intensities evenly.
* **Color overlay** — overlaying the constant color on top of the frame.
    * **Color** — a color that will be used to modulate the frame.
* **Custom code** — custom preprocessing code.
    * **Code** — Python code that will be used to process the frame.
        * **NOTE:** It provides a global 3D numpy array (height, width, RGB) called `input` and expects a processed array to be assigned to a global variable called `output`. `np`, `scipy`, and `skimage` modules are imported by default.
        * **WARNING:** Don't run an untrusted code.
* **Detailing** — an additional detailing pass that upscales the image and then scales it back, allowing for much higher precision at the cost of the processing speed.
    * **Scale** — upscaling factor.
        * **NOTE:** It doesn't affect the final output resolution, but rather the processing resolution itself.
    * **Sampling method** — same as the standard img2img option.
    * **Steps** — same as the standard img2img option.
    * **Denoising strength** — same as the standard img2img option.
* **Image overlay** — overlaying an image on top of the frame.
    * **Image** — an image that will be used to modulate the frame.
    * **Blurring** — blurring radius of the modulator image.
* **Interpolation** — blending of the current image towards the new image.
    * **Rate** — amount of blending between the current image and the new image (e.g. `0.5` will blend 50% of the new image into the current one).
* **Limiting** — limiting of the difference between the previous and the current image.
    * **Mode** — limiting mode.
        * **clamp** — clamp the difference, cutting off anything higher than **Maximum difference**.
        * **compress** — compress the difference, "squashing" its range to **Maximum difference**.
    * **Maximum difference** — maximum difference between the values of the individual color channels.
        * **NOTE:** This value represents both positive and negative values.
* **Measuring** — measuring various image values, such as mean luma or standard deviation among every color channel.
    * **Plot every N-th frame** — stride at which the plots will be rendered.
    * **NOTE:** Resulting plots will be placed into the `<project subdirectory>/session/metrics` directory.
* **Median** — averaging of neighboring pixels using the median filter.
    * **Radius** — averaging radius.
    * **Percentile** — percent at which the median value will be calculated; 0 — darkest, 100 — brightest.
* **Morphology** — processing of the frame using a morphological operation.
    * **Mode** — operation type.
        * **erosion** — makes the image details "thicker" and darker.
        * **dilation** — makes the image details "thinner" and brighter.
        * **opening** — erosion followed by dilation.
        * **closing** — dilation followed by erosion.
    * **Radius** — operation radius.
* **Noise compression** — basically, an actual algorithmical denoising, called noise compression to not be confused with the **img2img** denoising.
    * **Constant** — constant rate of the denoising. Generally should be very low, like 0.0002 or so, although it may vary.
    * **Adaptive** — adaptive rate of the denoising.
* **Noise overlay** — overlaying the random noise on top of the frame.
    * **Scale** — scale of the noise pattern; pixels.
    * **Octaves** — amount of progressively downscaled noise layers that will be mixed into a single one.
    * **Lacunarity** — downscale factor of each subsequent noise layer compared to a previous one.
    * **Persistence** — amplitude factor of each subsequent noise layer compared to a previous one.
    * **Seed** — static seed that will be used for generating the noise pattern.
    * **Use dynamic seed** — determines whether a currently processed frame's seed will be used or a filter's one.
* **Palettization** — applying a palette to the frame.
    * **Palette** — an image where _each_ pixel represents one color of a palette.
        * **NOTE:** Generally those images are very small (up to 256 pixels _total_) and contain just a few pixels representing the unique colors. For example, an 8x2 image contains 16 colors, and so on.
    * **Stretch** — enables linear stretching of the palette to fill all 256 colors, reducing the color banding.
        * **NOTE:** While it smoothes out the color transitions, it also introduces transitional tones that might not be intended in the palette.
    * **Dithering** — determines whether an image will be dithered in the process of quantization or not. In simple terms, it means reducing the color banding while using a limited color palette.
* **Processing** — the main Stable Diffusion processing procedure.
    * **NOTE:** Currently, all of the settings listed here are related to the averaging of several samples generated from a single frame.
    * **Sample count** — amount of samples to take for generating a frame.
        * **NOTE:** Reduces the jittering between the consecutive frames, increases the general middle-scale detail precision, multiplies amount of work to process each frame correspondingly, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** preprocessing effect).
    * **Trimming** — factor of distribution trimming (e.g. `0.25` trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
    * **Easing** — sample averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each subsequent sample; at the value of `0` all samples will be blended evenly.
        * **NOTE:** This parameter won't have any effect if **Trimming** is greater than `0`.
    * **Preference** — "brightness preference" of the averaging algorithm. At minimum, it prefers the darkest colors, at maximum—the brightest ones.
        * **NOTE:** The greater the **Trimming** is, the less this parameter will affect the result.
* **Saving** — automatic saving of resulting images.
    * **Save every N-th frame** — stride at which the frames will be saved.
    * **Archive mode** — disable saving of metadata inside of each frame (such as prompt, seed, etc.) and enable maximum compression.
* **Sharpening** — unsharp masking.
    * **Strength** — sharpening strength.
    * **Radius** — sharpening radius.
* **Symmetry** — makes the frame symmetrical.
    * **Horizontal** — symmetrize the horizontal axis.
    * **Vertical** — symmetrize the vertical axis.
* **Transformation** — geometric transformations applied to the entire frame.
    * **Translation X/Y** — amount of shifting to apply (e.g. X 0.3 — 30% of the image width).
    * **Rotation** — amount of rotation to apply; degrees.
    * **Scaling** — amount of scaling/zooming to apply (e.g. 2.0 — twice as large).
* **Video rendering** — automatic video rendering.
    * **Render draft/final every N-th frame** — stride at which a draft/final video will be rendered.
    * **Render draft/final on finish** — determines if a draft/final video will be rendered after finishing all iterations.
    * **NOTE:** All of the actual video configuration options are located in the **Video Rendering** tab.
