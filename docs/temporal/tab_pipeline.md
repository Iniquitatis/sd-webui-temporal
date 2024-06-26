**NOTE:** Almost all accordions within this tab can be reordered manually (using mouse or touchscreen, for example) to determine the order in which the modules will be invoked.  
**NOTE:** Order matters a lot. For example, applying **Noise overlay** before **Color correction** will make the noise color corrected, but otherwise the noise will be applied _on top_ of a color corrected image.  

All modules in this tab are marked with the following icons:
* ✨ — image filter that affects the image appearance directly.
* 🕓 — temporal module that takes multiple subsequent rendered frames into account in order to work. May take several iterations for the effect to be visible at all.
* 📈 — measuring module that measures various image values and builds corresponding graphs of the values' dynamics.
* 🛠 — tool module that doesn't directly affect an image, but rather does some action such as saving an image.
* 🧬 — neural network module that invokes Stable Diffusion in order to process an image.

* **Initial noise** — parameters of noise to use in a generation of the initial image if it's absent.
    * **Mode** — noise mode.
    * **Factor** — amount of noise that will be left unprocessed.
    * **Scale** — scale of the noise pattern; pixels.
    * **Detail** — amount of progressively downscaled noise layers that will be mixed into a single one.
    * **Lacunarity** — downscale factor of each subsequent noise layer compared to a previous one.
    * **Persistence** — amplitude factor of each subsequent noise layer compared to a previous one.
    * **Seed** — seed that will be used for generating the noise pattern.
    * **Use global seed** — determines whether the initial seed will be used or the one that's defined above.
* **Parallel** — amount of images to create/process in parallel.
* Generic options available to all modules:
    * **(✔️ to the left of a module name)** — determines if a module is enabled.
    * **(Eye icon to the right of a module name)** — determines if the module's results will be shown as a live preview.
* Generic options available to all filter modules:
    * **Amount** — amount of the processed image to be mixed in into the frame.
        * **Relative** — determines if the amount will be multiplied by the **img2img** denoising strength.
    * **Blend mode** — blending mode of the processed image.
    * **Mask** — an image that determines which areas will be processed by a filter; black—unprocessed, white—fully processed.
        * **Normalized** — normalization of the mask image to use the full range from black to white.
        * **Inverted** — inversion of the mask image.
        * **Blurring** — blurring radius of the mask image.
* Generic options available to all measuring modules:
    * **Plot every N-th frame** — stride at which the values will be measured.
    * **NOTE:** Resulting graphs will be placed into the `<project subdirectory>/metrics` directory.
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
    * **Image source** — an image source that will be used to match histograms. Simply put, an overall color balance of the frame will be matched against this image source.
    * **Normalize contrast** — normalize the contrast curve of the frame so that it's in range of 0.0–1.0.
    * **Equalize histogram** — equalize the image histogram, distributing the color intensities evenly.
* **Color level mean** — mean value measuring per RGB channel.
* **Color level sigma** — standard deviation measuring per RGB channel.
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
    * **Image source** — an image source that will be used to modulate the frame.
    * **Blurring** — blurring radius of the modulator image.
* **Interpolation** — interpolation of the current image towards the new image.
    * **Blending** — rate of introduction of colors from the new image.
    * **Movement** — rate of spatial shifting towards the similar areas of the new image.
    * **Radius** — radius of similar area detection.
* **Luminance mean** — luminance mean measuring.
* **Luminance sigma** — luminance standard deviation measuring.
* **Limiting** — limiting of the difference between the previous and the current image.
    * **Mode** — limiting mode.
        * **Clamp** — clamp the difference, cutting off anything higher than **Maximum difference**.
        * **Compress** — compress the difference, "squashing" its range to **Maximum difference**.
    * **Maximum difference** — maximum difference between the values of the individual color channels.
        * **NOTE:** This value represents both positive and negative values.
* **Median** — averaging of neighboring pixels using the median filter.
    * **Radius** — averaging radius.
    * **Percentile** — percent at which the median value will be calculated; 0 — darkest, 100 — brightest.
* **Morphology** — processing of the frame using a morphological operation.
    * **Mode** — operation type.
        * **Erosion** — makes the image details "thicker" and darker.
        * **Dilation** — makes the image details "thinner" and brighter.
        * **Opening** — erosion followed by dilation.
        * **Closing** — dilation followed by erosion.
    * **Radius** — operation radius.
* **Noise compression** — basically, an actual algorithmical denoising, called noise compression to not be confused with the **img2img** denoising.
    * **Constant** — constant rate of the denoising. Generally should be very low, like 0.0002 or so, although it may vary.
    * **Adaptive** — adaptive rate of the denoising.
* **Noise overlay** — overlaying the random noise on top of the frame.
    * **Mode** — noise mode.
    * **Scale** — scale of the noise pattern; pixels.
    * **Detail** — amount of progressively downscaled noise layers that will be mixed into a single one.
    * **Lacunarity** — downscale factor of each subsequent noise layer compared to a previous one.
    * **Persistence** — amplitude factor of each subsequent noise layer compared to a previous one.
    * **Seed** — static seed that will be used for generating the noise pattern.
    * **Use global seed** — determines whether a currently processed frame's seed will be used or a filter's one.
* **Noise sigma** — noise standard deviation measuring.
* **Palettization** — applying a palette to the frame.
    * **Palette** — an image where _each_ pixel represents one color of a palette.
        * **NOTE:** Generally those images are very small (up to 256 pixels _total_) and contain just a few pixels representing the unique colors. For example, an 8x2 image contains 16 colors, and so on.
    * **Stretch** — enables linear stretching of the palette to fill all 256 colors, reducing the color banding.
        * **NOTE:** While it smoothes out the color transitions, it also introduces transitional tones that might not be intended in the palette.
    * **Dithering** — determines whether an image will be dithered in the process of quantization or not. In simple terms, it means reducing the color banding while using a limited color palette.
* **Pixelization** — rounding to a specific virtual pixel size.
    * **Pixel size** — size of a virtual pixel. For example, at value of `8`, an image with resolution 1024x512 will _appear_ as if its resolution were 128x64—its actual resolution won't be affected.
* **Processing** — the main Stable Diffusion processing procedure.
    * **NOTE:** Currently, all of the settings listed here are related to the averaging of several samples generated from a single frame.
    * **Sample count** — amount of samples to take for generating a frame.
        * **NOTE:** Reduces the jittering between the consecutive frames, increases the general middle-scale detail precision, multiplies amount of work to process each frame correspondingly, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** preprocessing effect).
    * **Trimming** — factor of distribution trimming (e.g. `0.25` trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
    * **Easing** — sample averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each subsequent sample; at the value of `0` all samples will be blended evenly.
        * **NOTE:** This parameter won't have any effect if **Trimming** is greater than `0`.
    * **Preference** — "brightness preference" of the averaging algorithm. At minimum, it prefers the darkest colors, at maximum—the brightest ones.
        * **NOTE:** The greater the **Trimming** is, the less this parameter will affect the result.
* **Random sampling** — random picking of pixels from the new image.
    * **Chance** — chance of pixels from the new image to appear in the current image.
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
