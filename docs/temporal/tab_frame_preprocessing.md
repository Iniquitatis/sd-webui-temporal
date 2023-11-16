**NOTE:** Keep in mind that each option in this tab will be applied to _each_ subsequent frame before it will be fed to the inference engine. Given that this extension works as a loopback, it means that if you, for example, enable blurring, then it will be applied each time before Stable Diffusion does its magic. Therefore, if the blurring radius is too high, the resulting frames might start to get very blurry soon (which might be intended and preferable for some kind of artistic effect, of course).

* **Noise compression** — basically, an actual algorithmical denoising, called noise compression to not be confused with the **img2img** denoising.
    * **Enabled** — self-descriptive.
    * **Constant** — constant rate of the denoising. Generally should be very low, like 0.0002 or so, although it may vary.
    * **Adaptive** — adaptive rate of the denoising.
* **Color correction** — various parameters to keep frame's palette normalized.
    * **Enabled** — self-descriptive.
    * **Reference image** — an image that will be used to match histograms. Simply put, an overall color balance of the frame will be matched against this image.
    * **Normalize contrast** — normalize the contrast curve of the frame so that it's in range of 0.0–1.0.
* **Color balancing** — common color balancing.
    * **Enabled** — self-descriptive.
    * **Brightness** — target brightness of the frame.
    * **Contrast** — target contrast of the frame.
    * **Saturation** — target saturation of the frame.
* **Noise** — modulation of the frame by the random noise.
    * **Enabled** — self-descriptive.
    * **Amount** — amount of the noise to be mixed in into the frame.
        * **Relative** — determines if the amount will be multiplied by the **img2img** denoising strength.
    * **Mode** — blending mode of the noise.
    * **Mask** — an image that determines which areas will be processed by the effect; black—unprocessed, white—fully processed.
        * **Inverted** — inversion of the mask image.
        * **Blurring** — blurring radius of the mask image.
* **Modulation** — modulation of the frame by an image.
    * **Enabled** — self-descriptive.
    * **Amount** — amount of the modulator image to be mixed in into the frame.
        * **Relative** — determines if the amount will be multiplied by the **img2img** denoising strength.
    * **Mode** — blending mode of the modulator image.
    * **Image** — an image that will be used to modulate the frame.
    * **Blurring** — blurring radius of the modulator image.
    * **Mask** — an image that determines which areas will be processed by the effect; black—unprocessed, white—fully processed.
        * **Inverted** — inversion of the mask image.
        * **Blurring** — blurring radius of the mask image.
* **Tinting** — modulation of the frame by the constant color.
    * **Enabled** — self-descriptive.
    * **Amount** — amount of the color to be mixed in into the frame.
        * **Relative** — determines if the amount will be multiplied by the **img2img** denoising strength.
    * **Mode** — blending mode of the color.
    * **Color** — a color that will be used to modulate the frame.
    * **Mask** — an image that determines which areas will be processed by the effect; black—unprocessed, white—fully processed.
        * **Inverted** — inversion of the mask image.
        * **Blurring** — blurring radius of the mask image.
* **Sharpening** — unsharp masking.
    * **Enabled** — self-descriptive.
    * **Amount** — amount of the sharpening.
        * **Relative** — determines if the amount will be multiplied by the **img2img** denoising strength.
    * **Radius** — sharpening radius.
* **Transformation** — geometric transformations applied to the entire frame.
    * **Enabled** — self-descriptive.
    * **Translation X/Y** — amount of shifting to apply (e.g. X 0.3 — 30% of the image width).
    * **Rotation** — amount of rotation to apply; degrees.
    * **Scaling** — amount of scaling/zooming to apply (e.g. 2.0 — twice as large).
* **Symmetrize** — makes the frame symmetrical on the horizontal axis.
* **Blurring** — frame blurring.
    * **Enabled** — self-descriptive.
    * **Radius** — blurring radius.
* **Custom code** — custom preprocessing code.
    * **Enabled** — self-descriptive.
    * **Code** — Python code that will be used to process the frame.
        * **NOTE:** It provides a global 3D numpy array (height, width, RGB) called `input` and expects a processed array to be assigned to a global variable called `output`. `np`, `scipy`, and `skimage` modules are imported by default.
        * **WARNING:** Don't run an untrusted code.
