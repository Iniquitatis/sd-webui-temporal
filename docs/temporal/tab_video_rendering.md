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
* **Scaling** — video resolution upscaling/downscaling using Lanczos interpolation.
    * **Enabled** — self-descriptive.
    * **Width/Height** — final video resolution.
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
