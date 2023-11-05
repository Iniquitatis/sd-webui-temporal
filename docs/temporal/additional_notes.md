* Ancestral samplers are probably the only ones that might work on low denoising strengths, particularly the **Euler a**.
* Not every model/VAE can work correctly with the denoising strengths less than 0.5. Some of those might start to quickly darken the image, making it green/purple, or producing different artifacts. Those issues might be partially alleviated using the following tricks:
    * **Settings/Sampler parameters/Always discard next-to-last sigma** tends to reduce the possibility of a model to go crazy.
    * Using the `vae-ft-mse-840000-ema-pruned.safetensors` instead of the model's one can reduce the color balance issues.
    * Using the **Frame Preprocessing/Color correction/Image** to correct the histogram to that image.
    * Applying any kind of frame preprocessing drives an image farther from what Stable Diffusion produces, so in some cases might also help.
* Keep an eye on the space on your HDD, as thousands of just 512x512 images might fill it up quickly.
* **General/Start from scratch** + **General/Save session** combo is useful when experimenting/tinkering with the parameters before starting the longer rendering process.
