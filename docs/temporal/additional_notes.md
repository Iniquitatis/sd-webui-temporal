* Ancestral samplers are probably the only ones that might work on low denoising strengths, particularly **Euler a**.
* Not every model/VAE can work correctly with the denoising strengths less than 0.5. Some of those might soon start to darken the resulting frames, making them green/purple, or producing different artifacts. Those issues might be partially alleviated using the following tricks:
    * Setting the **Settings/Sampler parameters/Always discard next-to-last sigma** tends to reduce the possibility of a model to go crazy.
    * Using the `vae-ft-mse-840000-ema-pruned.safetensors` instead of the model's one can reduce the color balance issues.
    * Using the **Frame Preprocessing/Color correction/Reference image** to match the histogram against that image. It might also make all of the resulting frames more-or-less consistent palette-wise, which might or might not be preferable.
    * Applying any kind of frame preprocessing drives an image farther from what Stable Diffusion produces, so in some cases might also help.
* Keep an eye on the space on your HDD, as thousands of just 512x512 images might fill it up quickly.
* **General/Start from scratch** + **General/Save session** combo is useful when experimenting/tinkering with the parameters before starting the longer rendering process.
* ControlNet depth maps can be used as masks inside the preprocessing tab's effects. For example, applying a blurring masked by a depth map can create sort of a depth of field.
