# Stable Diffusion Web UI Temporal Extension
Basically, it's a loopback on steroids that allows doing something like [this](https://youtu.be/Q4gowAKcDNo) (requires `ffmpeg` to be installed globally), testing how models behave in a large loop, and generating pictures that can't be generated easily under normal circumstances.

**IMPORTANT:** This extension is still at its early stage, so no backwards compatibility is guaranteed as of now.



## Usage
1. Go to the **img2img** tab.
2. Select **Temporal** in the **Script** dropdown menu.
3. Set sampler, steps, resolution, etc. as usual.
    * **NOTE:** According to my observations, it's preferable to enable the **Always discard next-to-last sigma** option in the sampler options in order to reduce the possibility of a model to go crazy on low denoising levels (like 0.2 or lower).
4. Change the extension settings to your own liking (should be self-descriptive).
    * **NOTE:** Keep in mind that the frame preprocessing (**Frame Preprocessing** tab) will be applied to _each_ subsequent frame. Given that this extension works as a loopback, it means that if you, for example, enable blurring, then it will be applied each time before Stable Diffusion does its magic. Therefore, if the blurring radius is too high, the resulting frames might start to get very blurry soon (which might be intended and preferable for some kind of artistic effect, of course).
5. Hit **Generate**.
6. Grab yourself some coffee and get ready to wait.
    * While waiting, you can go to the extension's **Video Rendering** tab and hit **Render draft/final** that will start rendering a video simultaneously with the usual SD generation process. Be sure to note that doing that simultaneously might stress your system quite a lot.



## Roadmap
Nothing in particular, aside from fixing bugs and adding a feature here and there. I'm using this extension almost all the time, so it can be considered—contrary to the note above—feature-complete. At least until I find something I'm missing, that is.



## FAQ
**Q:** Why does this extension even exist if there's already `<insert extension name here>`?  
**A:** To be honest, just a typical thing where I'm developing something for my own needs with a secondary intent to release it one day, but never releasing it afterwards. Then, I haven't used a lot of extensions for Web UI, and I've learned that there are similar things after I've started developing this one. So, no big reason. Just for fun and probably some extra control.



## License
See LICENSE file in the root of this repository.
