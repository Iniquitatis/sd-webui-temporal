**NOTE:** Each option in this tab will be applied to the frame before it will be fed to the inference engine.

* **Noise compression** — basically, an actual algorithmical denoising, called noise compression to not be confused with the **img2img** denoising.
    * **Enabled** — self-descriptive.
    * **Constant** — constant rate of the denoising. Generally should be very low, like 0.0002 or so, although it may vary.
    * **Adaptive** — adaptive rate of the denoising.
* **Color correction** — various parameters to keep frame's palette normalized.
    * **Enabled** — self-descriptive.
    * **Reference image** — an image that will be used to match histograms. Simply put, an overall color balance of the frame will be matched against this image.
    * **Normalize contrast** — normalize a contrast curve of the frame so that it's in range of 0.0–1.0.
* **Color balancing** — common color balancing.
    * **Enabled** — self-descriptive.
    * **Brightness** — target brightness of the frame.
    * **Contrast** — target contrast of the frame.
    * **Saturation** — target saturation of the frame.
* **Noise** — modulation of the frame by the random noise.
    * **Enabled** — self-descriptive.
    * **Amount** — amount of the noise to be mixed in into the frame.
        * **Relative** — determines if the amount will be multiplied by **img2img** denoising strength.
    * **Mode** — blending mode of the noise.
* **Modulation** — modulation of the frame by the random noise.
    * **Enabled** — self-descriptive.
    * **Amount** — amount of the image to be mixed in into the frame.
        * **Relative** — determines if the amount will be multiplied by **img2img** denoising strength.
    * **Mode** — blending mode of the image.
    * **Image** — an image that will be used to modulate the frame.
    * **Blurring** — radius of blurring of an image.
* **Tinting** — modulation of the frame by the constant color.
    * **Enabled** — self-descriptive.
    * **Amount** — amount of the color to be mixed in into the frame.
        * **Relative** — determines if the amount will be multiplied by **img2img** denoising strength.
    * **Mode** — blending mode of the color.
    * **Color** — a color that will be used to modulate the frame.
* **Sharpening** — unsharp masking.
    * **Enabled** — self-descriptive.
    * **Amount** — amount of the sharpening.
        * **Relative** — determines if the amount will be multiplied by **img2img** denoising strength.
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
    * **Code** — self-descriptive.
