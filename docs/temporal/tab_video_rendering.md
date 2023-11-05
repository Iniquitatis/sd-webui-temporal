**NOTE:** Resulting videos will be placed in the output directory.  
**NOTE:** Drafts will be rendered without the frame upscaling (width/height will be exactly the same as the rendered frames), interpolation, or deflickering, leaving only the general structure of the resulting video intact and making rendering much faster.

* **Width/Height** — upscaled video resolution. Lanczos interpolation is used to upscale frames if they don't match this resolution.
* **Frames per second** — virtual framerate. It corresponds to how often the frames change, but not necessarily to the actual video framerate that is a subject to change by enabling the interpolation.
* **Interpolation** — upscales the resulting video framerate to 60 frames per second using the motion interpolation algorithm in order to keep the frame transitions smooth.
* **Motion blur subframes** — additional subframe count that might make the resulting video even smoother. Results are mostly negligible, and each subframe multiplies amount of work by the factor of `x + 1`.
* **Deflickering** — applies a deflickering algorithm, reducing the luminance variations between frames.
* **Looping** — makes the resulting video loop in a "boomerang"-like fashion (e.g. `1 2 3 4 3 2 1`).
* **Frame number overlay** — enables printing of the frame number in the top-left corner of the resulting video. Used mostly for tinkering or debugging.
* **Render draft/final when finished** — automatically start the video rendering after all frames have been rendered.
* **Render draft/final** — start the video rendering immediately.
* **Preview** — a video player that will show the result after the _manually started_ video rendering finishes.
