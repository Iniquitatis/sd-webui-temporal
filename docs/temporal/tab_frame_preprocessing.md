**NOTE:** Keep in mind that each option in this tab will be applied to _each_ subsequent frame before it will be fed to the inference engine. Given that this extension works as a loopback, it means that if you, for example, enable blurring, then it will be applied each time before Stable Diffusion does its magic. Therefore, if the blurring radius is too high, the resulting frames might start to get very blurry soon (which might be intended and preferable for some kind of artistic effect, of course).

* **Order** — order in which the preprocessing effects will be applied. Every effect that is not listed will be applied after the listed ones in the alphabetical order.
    * **NOTE:** Order matters a lot. Applying, for example, noise before tinting will make the noise tinted, but otherwise the noise will be applied on top of a tinted image.
* Generic options, available to all preprocessing effects:
    * **Enabled** — determines whether a preprocessing effect is turned on or not.
    * **Amount** — amount of the processed image to be mixed in into the frame.
        * **Relative** — determines if the amount will be multiplied by the **img2img** denoising strength.
    * **Mode** — blending mode of the processed image.
    * **Mask** — an image that determines which areas will be processed by a preprocessing effect; black—unprocessed, white—fully processed.
        * **Normalized** — normalization of the mask image to use the full range from black to white.
        * **Inverted** — inversion of the mask image.
        * **Blurring** — blurring radius of the mask image.
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
* **Image overlay** — overlaying an image on top of the frame.
    * **Image** — an image that will be used to modulate the frame.
    * **Blurring** — blurring radius of the modulator image.
* **Median** — averaging of neighboring pixels using the median filter.
    * **Radius** — averaging radius.
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
* **Noise ovleray** — overlaying the random noise on top of the frame.
    * **Scale** — scale of the noise pattern; pixels.
    * **Octaves** — amount of progressively downscaled noise layers that will be mixed into a single one.
    * **Lacunarity** — downscale factor of each subsequent noise layer compared to a previous one.
    * **Persistence** — amplitude factor of each subsequent noise layer compared to a previous one.
    * **Seed** — static seed that will be used for generating the noise pattern.
    * **Use dynamic seed** — determines whether a currently processed frame's seed will be used or an effect's one.
* **Palettization** — applying a palette to the frame.
    * **Palette** — an image where _each_ pixel represents one color of a palette.
        * **NOTE:** Generally those images are very small (up to 256 pixels _total_) and contain just a few pixels representing the unique colors. For example, an 8x2 image contains 16 colors, and so on.
    * **Stretch** — enables linear stretching of the palette to fill all 256 colors, reducing the color banding.
        * **NOTE:** While it smoothes out the color transitions, it also introduces transitional tones that might not be intended in the palette.
    * **Dithering** — determines whether an image will be dithered in the process of quantization or not. In simple terms, it means reducing the color banding while using a limited color palette.
* **Sharpening** — unsharp masking.
    * **Strength** — sharpening strength.
    * **Radius** — sharpening radius.
* **Symmetry** — makes the frame symmetrical on the horizontal axis.
* **Transformation** — geometric transformations applied to the entire frame.
    * **Translation X/Y** — amount of shifting to apply (e.g. X 0.3 — 30% of the image width).
    * **Rotation** — amount of rotation to apply; degrees.
    * **Scaling** — amount of scaling/zooming to apply (e.g. 2.0 — twice as large).
