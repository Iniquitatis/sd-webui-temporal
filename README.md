# Stable Diffusion Web UI Temporal Extension
Basically, it's a loopback on steroids that allows doing something like [this](https://youtu.be/h_nXV1IhWMY) (requires `ffmpeg` to be installed globally), testing how models behave in a large loop, and generating pictures that can't be generated easily under normal circumstances.

**IMPORTANT:** This extension is still at its early stage, so no backwards compatibility is guaranteed as of now.



## Usage
1. Go to the **img2img** tab.
2. _(Optional)_ Set the **img2img** image. If it's not set, it will be generated automatically according to the provided settings, similarly to how **txt2img** does this.
3. Set the sampling method/steps, image resolution, etc.
4. Select **Temporal** in the **Script** dropdown.
5. Change the extension settings to your own liking.
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
