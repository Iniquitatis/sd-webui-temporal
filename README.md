# Stable Diffusion Web UI Temporal Extension
Basically, it's a loopback on steroids that allows doing something like [this](https://youtu.be/h_nXV1IhWMY) (requires `ffmpeg` to be installed globally), testing how models behave in a large loop, and generating pictures that can't be generated easily by normal means.

**IMPORTANT:** This extension is still at its early stage, so no backwards compatibility is guaranteed as of now.



## Preprocessing  
The most important feature of the extension is its automatic image preprocessing system that invokes each time before sending an image further down the loop.  
The interesting thing about such an approach is that Stable Diffusion adapts the composition to every possible change in the image, even the slightest one, so, for example, if a slight blue tinting is added to the image on each iteration, then the resulting image/video will lean towards the actual night scenes, not just the darkened/tinted ones. Therefore, each step of preprocessing "nudges" each iteration towards some specific outcome and changes the final result (probably not as noticeable in the beginning, but drastically after a lot of iterations)—just a single seed can look very differently depending on the used parameters.



## Observations
There are a number of peculiarities about the loopback approach:
* It can help to fix the image composition: making limbs more proportional, removing extra fingers, making facial features more precise, and so on.
* The lower denoising strength is, the more fine details will be added on each iteration. It might be preferable in the beginning, but after, say, several hundred iterations a model might start to get "crazy", producing weird oversaturated lines and dots, and then even drifting further away from the original prompt. This issue can be partially alleviated by using parts of the preprocessing system (can't be specific here, as it works case-by-case, depending on the current model/VAE).
* The one VAE that arguably produces the least amount of artifacts is the `vae-ft-mse-840000-ema-pruned`. It's not artifact-proof, but at the very least it doesn't tend to shift the color palette to the green/purple/darkened/overly contrasted colors.
* Usually, applying the color correction image (1.0) and modulation image (0.05-0.15 for denoising strength of 0.05) can help to stabilize the color palette and the overall composition, although it will reduce the amount of changes in the image. ControlNet can also be used for that.
* Low denoising strength (~0.05) along with the "frame merging" feature (~20) can be used as a "style changer" of sorts. Note that it will make the resulting image/video slightly blurrier, removing the fine details, such as skin pores, individual hair strands, and so on. Of course, those or similar settings can be also used if just a very slow morphing effect is needed.

_If you use the extension, then please report your observations: I can't test every possible model/VAE/combination of parameters myself, especially on the fairly weak hardware (generating thousands of frames takes quite some time)._



## Usage
1. Go to the **img2img** tab.
2. _(Optional)_ Set the **img2img** image. If it's not set, it will be generated automatically according to the provided settings, similarly to how **txt2img** does this.
3. Set the sampling method/steps, image resolution, etc.
4. Select **Temporal** in the **Script** dropdown.
5. Change the extension settings to your own liking. Tinkering is encouraged.
    * **NOTE:** Consult the **Help** tab if you're stuck somewhere in the process of tinkering, especially if you use the extension for the first time.
6. Hit **Generate**.
7. Grab yourself some coffee and get ready to wait.
    * While waiting, you can go to the extension's **Video Rendering** tab and hit **Render draft/final** that will start rendering a video simultaneously with the usual SD generation process. Be sure to note that doing that simultaneously might stress your system quite a lot.



## Roadmap
Nothing in particular, aside from fixing bugs and adding a feature here and there. I'm using this extension almost all the time, so it can be considered—contrary to the note above—feature-complete. At least until I find something I'm missing, that is.



## FAQ
**Q:** Why does this extension even exist if there's already `<insert extension name here>`?  
**A:** To be honest, just a typical thing where I'm developing something for my own needs with a secondary intent to release it one day, but never releasing it afterwards. Then, I haven't used a lot of extensions for Web UI, and I've learned that there are similar things after I've started developing this one. So, no big reason. Just for fun and probably some extra control.



## License
See LICENSE file in the root of this repository.
