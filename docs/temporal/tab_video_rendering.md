**NOTE:** Resulting videos will be placed in the output directory.  
**NOTE:** Draft mode simply skips the deflickering, interpolation, and scaling steps, making the rendering process much faster.

* **Frames per second** — virtual framerate. It corresponds to how often the frames change, but not necessarily to the actual video framerate, which is still a subject to change by the interpolation.
* **Looping** — makes the resulting video loop in a "boomerang"-like fashion (e.g. `1 2 3 4 3 2 1`).
* **Deflickering** — reduces the luminance variations between frames.
    * **Enabled** — self-descriptive.
    * **Frames** — amount of frames to take into account when calculating the mean luminance.
* **Interpolation** — video framerate upscaling/downscaling using motion interpolation in order to keep the transitions between frames smooth.
    * **Enabled** — self-descriptive.
    * **Frames per second** — final video framerate.
    * **Motion blur subframes** — additional subframe count that might make the resulting video even smoother. Results are mostly negligible, and each subframe multiplies the amount of work by the factor of `x + 1`.
* **Temporal blurring** — weighted averaging of several consecutive frames.
    * **Enabled** — self-descriptive.
    * **Radius** — kernel radius; total amount of averaged frames equals to `x * 2 + 1`.
    * **Easing** — kernel easing factor.
        * Value of 0 means that every frame will be averaged in an equal proportion, whereas value greater than 0 makes a distribution ranging from 0 to 1.
        * Value greater than 0 makes a soft distribution curve.
        * Value of 1 makes a triangle distribution curve.
        * Value greater than 1 makes a sharp distribution curve.
    * **EXAMPLES:**
        * `Radius: 1; Easing: 1.0 = Weights [0.5 1 0.5]`
        * `Radius: 1; Easing: 0.5 = Weights [0.707 1 0.707]`
        * `Radius: 1; Easing: 2.0 = Weights [0.25 1 0.25]`
        * `Radius: 3; Easing: 0.0 = Weights [1 1 1 1 1 1 1]`
* **Color balancing** — common color balancing.
    * **Brightness** — target brightness level.
    * **Contrast** — target contrast level.
    * **Saturation** — target saturation level.
* **Sharpening** — unsharp masking.
    * **Enabled** — self-descriptive.
    * **Strength** — sharpening strength.
    * **Radius** — sharpening radius.
* **Chromatic aberration** — a fake chromatic aberration-like effect that shifts red and blue channels away from the pixel's center.
    * **Enabled** — self-descriptive.
    * **Distance** — distance of channel shifting in pixels.
* **Scaling** — video resolution upscaling/downscaling using Lanczos interpolation.
    * **Enabled** — self-descriptive.
    * **Width/Height** — final video resolution.
    * **Padded** — pad video with black borders if the aspect ratio doesn't match, otherwise simply stretch the frames.
* **Frame number overlay** — enables printing the frame number in the top-left corner of the resulting video.
    * **Enabled** — self-descriptive.
    * **Font size** — self-descriptive.
    * **Text color** — self-descriptive.
    * **Text alpha** — self-descriptive.
    * **Shadow color** — self-descriptive.
    * **Shadow alpha** — self-descriptive.
* **Render draft/final when finished** — automatically start the video rendering after all frames have been rendered.
* **Render draft/final** — start the video rendering immediately.
* **Preview** — a video player that will show the result after the _manually started_ video rendering finishes.
