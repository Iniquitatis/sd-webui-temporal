**NOTE:** Resulting videos will be placed in the output directory.

* **Width/height** — video resolution. Uses Lanczos interpolation to upscale frames if they don't match this resolution.
* **Frames per second** — self-descriptive.
* **Interpolation** — upscales or downscales the video framerate to 60 frames per second using motion interpolation algorithm in order to keep frame transitions smooth.
* **Motion blur subframes** — additional subframe count that might make a resulting video even smoother. Results are mostly negligible, and each subframe multiplies amount of work by the factor of `x + 1`.
* **Deflickering** — applies a deflickering algorithm.
* **Looping** — makes video looping in a "boomerang"-like fashion (e.g. `1 2 3 4 3 2 1`).
* **Frame number overlay** — enables printing of the frame number in the top-left corner of the resulting video. Used mostly for tinkering or debugging.
* **Render draft/final when finished** — automatically start the video rendering after all frames have been rendered.
* **Render draft/final** — start the video rendering immediately.
* **Preview** — video player that will show the results after the _manually started_ video rendering finishes.
