# Stable Diffusion Web UI Temporal Extension
This extension turns the text-to-image/image-to-image process into less of a cooking and more of a baking. Basically, it's a loopback on steroids that allows doing something like [this](https://youtu.be/h_nXV1IhWMY): that is, rendering sequences of images that get processed automatically given the user-defined algorithm and then sent further down the loop.  
And yes, it can be used as the text-to-image generator (not that it's technically very different from image-to-image in any case), despite being available only in the **img2img** tab.



## Pipeline
The main point of the extension is that Stable Diffusion adapts the composition to every possible change in an image—even the slightest one—so, for example, if an image gets slightly tinted with the dark blue color on each iteration, then the resulting images will start leaning towards the actual night scenes instead of being just darkened/tinted. Therefore, each pipeline pass over an image "nudges" the previous image towards some specific outcome and changes the final result (probably not as noticeable in the beginning, but drastically after a lot of iterations)—just a single seed can look very differently depending on the used parameters.



## Observations
There are a number of peculiarities about the loopback approach:
* It can (and most probably will) help to fix the image composition: making limbs more proportional, making facial features more precise, shifting the reflected lighting to come from the proper light sources, and so on.
* Oftentimes, the **Color correction** module has to be enabled, because Stable Diffusion itself tends to shift the color palette towards some specific color: For example, some VAEs of SD 1.5 make an image more green, while others make it more purple, and SDXL by default makes it more yellow. Not to mention, this module might help to set the entire "mood" of the composition.
* Given that each previous image in the loop serves as the "context" for the next image, some of the "temporal" modules (**Averaging**, **Interpolation**, etc.) may make this contextual information even more precise when put somewhere before the Stable Diffusion invocation (they can obviously be put after that, but that'll give a different effect).
* Denoising strength less than `0.5` is generally discouraged. Tests on SD 1.5 have shown that while it can add far more fine details on each iteration, it also often tends to go completely crazy after some time, producing weird oversaturated lines, dots, and even drifting further away from the prompt. Moreover, it's heavily sampler/VAE-dependent.

_If you use the extension, please report your observations: I can't test every possible model/VAE/parameter combination myself, especially on the fairly weak hardware (generating thousands of frames takes quite some time)._



## Prerequisites
* _(Optional)_ [FFmpeg](https://ffmpeg.org/) system-wide installation. If it's not installed, then no video-related functionality will be available.



## Usage
1. Go to the **img2img** tab.
2. Set the parameters according to the [Example](#example) section. Tinkering is encouraged.
    * **NOTE:** Consult the **Help** tab if you're stuck somewhere in the process of tinkering, especially if you use the extension for the first time.
3. Hit **Generate**.
4. Grab yourself some coffee and get ready to wait.



## Example
Here is the very simple set of parameters to help you quickly check out the extension:
* **Prompt:** (anything)
* **Negative prompt:** (anything)
* _(Optional)_ **Image:** (anything)
    * **NOTE:** If it's not set, it will be generated automatically according to the provided parameters, similarly to how **txt2img** does this.
* **Sampling method:** `DPM++ 2M Karras`/`Euler a`
* **Sampling steps:** `16`
* **Width:** (anything)
* **Height:** (anything)
* **CFG scale:** `7`
* **Denoising strength:** `0.5`
* **Script:** `Temporal`
    * **General** tab
        * **Project:** (any name—this one will represent a project's directory name)
        * **Load session:** `disabled`
        * **Continue from last frame:** `disabled`
        * **Iteration count:** (any number)
    * **Pipeline** tab
        * (Put the next three modules at top by dragging them.)
        * **Processing:** `enabled`
        * **Color correction:** `enabled`
            * **Image source: Type:** `Image`
            * **Image source: Image:** (any image to grab the palette from)
        * **Saving:** `enabled`



## Roadmap
Nothing in particular, aside from fixing bugs and adding a feature here and there. I'm using this extension almost all the time (more than half of a year, in fact), so it can be considered feature-complete. At least until I find something I'm missing, that is.



## FAQ
**Q:** Why does this extension even exist if there's already `<insert extension name here>`?  
**A:** To be honest, just a typical thing where I'm developing something for my own needs with a secondary intent to release it one day, but never releasing it afterwards. Then, I haven't used a lot of extensions for Web UI, and I've learned that there are similar things after I've started developing this one. So, no big reason. Just for fun and probably some extra control.



## License
See LICENSE file in the root of this repository.
