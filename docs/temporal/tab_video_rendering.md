**NOTE:** Resulting videos will be placed in the output directory.  
**NOTE:** Draft mode simply skips the deflickering, interpolation, and scaling steps, making the rendering process much faster.

* **Frames per second** — virtual framerate. It corresponds to how often the frames change, but not necessarily to the actual video framerate, which is still a subject to change by the interpolation.
* **Looping** — makes the resulting video loop in a "boomerang"-like fashion (e.g. `1 2 3 4 3 2 1`).
* **Order** — order in which the video filters will be applied. Every filter that is not listed will be applied after the listed ones in the alphabetical order.
    * **NOTE:** Order matters a lot. Applying, for example, sharpening before frame number overlay will make the text sharpened, but otherwise the text will be drawn on top of a sharpened video.
* Generic options, available to all video filters:
    * **Enabled** — determines whether a filter is turned on or not.
* **Chromatic aberration** — a fake chromatic aberration-like effect that shifts red and blue channels away from the pixel's center.
    * **Distance** — distance of channel shifting in pixels.
* **Color balancing** — common color balancing.
    * **Brightness** — target brightness level.
    * **Contrast** — target contrast level.
    * **Saturation** — target saturation level.
* **Deflickering** — reduces the luminance variations between frames.
    * **Frames** — amount of frames to take into account when calculating the mean luminance.
* **Interpolation** — video framerate upscaling/downscaling using motion interpolation in order to keep the transitions between frames smooth.
    * **Frames per second** — interpolated video framerate.
    * **Motion blur subframes** — additional subframe count that might make the resulting video even smoother. Results are mostly negligible, and each subframe multiplies the amount of work by the factor of `x + 1`.
* **Scaling** — video resolution upscaling/downscaling using Lanczos interpolation.
    * **Width/Height** — final video resolution.
    * **Padded** — pad video with black borders if the aspect ratio doesn't match, otherwise simply stretch the frames.
* **Sharpening** — unsharp masking.
    * **Strength** — sharpening strength.
    * **Radius** — sharpening radius.
* **Temporal averaging** — averaging of several consecutive frames.
    * **Radius** — filter radius; total amount of averaged frames equals to `x * 2 + 1`.
    * **Algorithm** — algorithm to use when computing the average.
        * **mean** — produces blurry video.
        * **median** — produces sharper video than **mean**, but more prone to artifacts.
    * **Easing** — kernel easing factor.
        * **NOTE:** This parameter is relevant only for **mean** algorithm.
        * Value of 0 means that every frame will be averaged in an equal proportion, whereas value greater than 0 makes a distribution ranging from 0 to 1.
        * Value greater than 0 makes a soft distribution curve.
        * Value of 1 makes a triangle distribution curve.
        * Value greater than 1 makes a sharp distribution curve.
        * **EXAMPLES:**
            * `Radius: 1; Easing: 1.0 = Weights [0.5 1 0.5]`
            * `Radius: 1; Easing: 0.5 = Weights [0.707 1 0.707]`
            * `Radius: 1; Easing: 2.0 = Weights [0.25 1 0.25]`
            * `Radius: 3; Easing: 0.0 = Weights [1 1 1 1 1 1 1]`
* **Text overlay** — draws the text.
    * **Text** — text that will be drawn. Variables should be enclosed between `{}`. Available variables are:
        * **frame** — number of the currently shown frame.
    * **Anchor X/Y** — anchor of where the text should be placed regarding the frame borders. 0.0 — left/top, 0.5 — center/center, 1.0 — right/bottom.
    * **Offset X/Y** — position of the text relative to the anchor; pixels.
    * **Font** — name of the system-installed font.
    * **Font size** — size of the font; pixels.
    * **Text color** — color of the text.
    * **Text alpha** — opacity of the text.
    * **Shadow offset X/Y** — offset of the text shadow; pixels.
    * **Shadow color** — color of the text shadow.
    * **Shadow alpha** — opacity of the text shadow.
* **Render draft/final when finished** — automatically start the video rendering after all frames have been rendered.
* **Render draft/final** — start the video rendering immediately.
* **Preview** — a video player that will show the result after the _manually started_ video rendering finishes.
