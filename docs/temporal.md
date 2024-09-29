# Main User Interface

## Preset
The name of a currently selected or non-existent preset. If a non-existent name is entered here, a new preset will be created after pressing 💾 button.

### 🔄
Update the preset list.

### 📂
Load the currently selected preset.

### 💾
Save the currently selected preset.

### ✏️
Rename the currently selected preset.

### 🗑️
Delete the currently selected preset.

## Project
The name of a currently selected or non-existent project. If a non-existent name is entered here, a new project will be created after pressing the **Generate** button.

**NOTE:** Due to some technical limitations, pressing 🔄 button is required to make the new project appear in the list.

### 🔄
Update the project list.

### 📂
Load the UI parameters from the currently selected project.

### ✏️
Rename the currently selected project.

### 🗑️
Delete the currently selected project.

**NOTE:** If ControlNet extension is installed, its parameters will also be saved alongside the project data.



# General Tab

## Load parameters
Determines if the generation parameters will be taken from the currently selected project (if it exists) or from the UI.

## Continue from last frame
Determines if the generation will be continued from the last iteration of the currently selected project or restarted from scratch, discarding all of the previously rendered images.

## Iteration count
Amount of iterations that will be performed in the current session.

**NOTE:** This is _not_ a total amount of rendered frames. For example, if a project already has 57 frames rendered, then another `Iteration count` frames will be rendered before automatically stopping the process.



# Information Tab

## Description
Textual description of the currently selected project.

## Gallery
Preview of the rendered images from the currently selected project.

## Page
Currently selected page of the **Gallery**.

## Parallel
Index of an image set to preview in the **Gallery**.



# Pipeline Tab

## Initial noise
Noise that will be used as a base in the generation of the initial image (if it's absent).

### Mode
Mode of the noise generator that determines the shape of the resulting noise.

### Factor
Amount of noise that will be left unprocessed.

### Scale
Scale of the noise pattern, measured in pixels.

### Detail
Amount of progressively downscaled noise layers that will be mixed into to form the final noise.

### Lacunarity
Downscale factor of each subsequent noise layer compared to a previous one.

### Persistence
Amplitude factor of each subsequent noise layer compared to a previous one.

### Seed
An arbitrary value that will be used for generating the noise pattern.

### Use global seed
Determines if the initial seed will be used or the one that's defined above.

## Parallel
Amount of images to create/process in parallel.

## (A bunch of accordions)
See the [Pipeline Modules](#pipeline-modules) section.

**NOTE:** All of these accordions can be reordered manually (using mouse or touchscreen, for example) to determine the order in which the modules will be invoked.



# Pipeline Modules
These modules represent the basic building blocks of a pipeline.

**NOTE:** Order matters a lot. For example, applying **🖌️ Noise** before **✨ Color correction** will make the noise color corrected, but otherwise the noise will be applied _on top_ of a color corrected image.

## Module categories
All modules in this tab are marked with icons to denote the category to which they belong.

### ✨ — Image filter
Image filter that affects the image appearance directly.

### 🕓 — Temporal module
Temporal module that takes multiple subsequent rendered frames into account in order to work. May take several iterations for the effect to be visible at all.

### 📈 — Measuring module
Measuring module that measures various image values and builds corresponding graphs of the values' dynamics.

### 🖌️ — Painting module
Painting module that draws something on top of the currently processed image.

### 🛠 — Tool module
Tool module that doesn't directly affect an image, but rather does some action such as saving an image.

### 🧬 — Neural network module
Neural network module that invokes Stable Diffusion in order to process an image.

## Generic options

### (✔️ to the left of a module name)
Determines if a module is enabled.

### (Eye icon to the right of a module name)
Determines if the module's results will be shown as a live preview.

## Generic image filter options

### Amount
Amount of the processed image to be mixed in into the frame.

### Relative
Determines if the amount will be multiplied by the **img2img** denoising strength.

### Blend mode
Blending mode of the processed image.

### Mask: Image
An image that determines which areas will be processed by a filter; black—unprocessed, white—fully processed.

### Mask: Normalized
Normalization of the mask image to use the full range from black to white.

### Mask: Inverted
Inversion of the mask image.

### Mask: Blurring
Blurring radius of the mask image.

## Generic measuring module options

### Plot every N-th frame
Stride at which the values will be measured.

**NOTE:** Resulting graphs will be placed into the `<project subdirectory>/metrics` directory.

## Blurring
Gaussian blurring of an image.

### Radius
Blurring radius.

## Color balancing
Common color balancing.

**NOTE:** This filter is idempotent.

### Brightness
Desired image brightness level.

### Contrast
Desired image contrast level.

### Saturation
Desired image saturation level.

## Color correction
Various color correction options.

### Image source
An image source that will be used to match histograms. Simply put, an overall color balance of the frame will be matched against this image source.

### Normalize contrast
Normalize the contrast curve of the frame so that it's in range of 0.0–1.0.

### Equalize histogram
Equalize the image histogram, distributing the color intensities evenly.

## Color overlay
Overlaying the constant color on top of the frame.

### Color
A color that will overlaid on top of an image.

## Custom code
Custom preprocessing code.

**WARNING:** Don't run an untrusted code.

### Code
Python code that will be used to process the frame.

**NOTE:** It provides a global 3D numpy array (height, width, RGB) called `input` and expects a processed array to be assigned to a global variable called `output`. `np`, `scipy`, and `skimage` modules are imported by default.

## Image overlay
Overlaying of an arbitrary image on top of the currently processed image.

### Image source
An image source that provides an image that will be overlaid on top of the currently processed image.

### Blurring
Blurring radius of the overlaid image.

## Median
Averaging of neighboring pixels using the median filter.

### Radius
Averaging radius.

### Percentile
Percent at which the median value will be calculated. `0` — darkest of the neighbors, `100` — brightest of the neighbors.

## Morphology
Processing of the image using a morphological operation.

### Mode
Operation type.

* **Erosion** — makes the image details "thicker" and darker.
* **Dilation** — makes the image details "thinner" and brighter.
* **Opening** — erosion followed by dilation.
* **Closing** — dilation followed by erosion.

### Radius
Operation radius.

## Noise compression
Basically, an actual algorithmical denoising, called noise compression to not be confused with the **img2img** denoising.

### Constant
Constant rate of the denoising. Generally should be very low, like `0.0002` or so, although it may vary.

### Adaptive
Adaptive rate of the denoising.

## Noise overlay
Overlaying of the value noise on top of the currently processed image.

### Mode
Noise mode.

### Scale
Scale of the noise pattern measured in pixels.

### Detail
Amount of progressively downscaled noise layers that will be mixed into a single one.

### Lacunarity
Downscale factor of each subsequent noise layer compared to a previous one.

### Persistence
Amplitude factor of each subsequent noise layer compared to a previous one.

### Seed
Static seed that will be used for generating the noise pattern.

### Use global seed
Determines whether a currently processed images's seed will be used or a filter's one.

# TODO

* **Palettization** — applying a palette to the frame.
    * **Palette** — an image where _each_ pixel represents one color of a palette.
        * **NOTE:** Generally those images are very small (up to 256 pixels _total_) and contain just a few pixels representing the unique colors. For example, an 8x2 image contains 16 colors, and so on.
    * **Stretch** — enables linear stretching of the palette to fill all 256 colors, reducing the color banding.
        * **NOTE:** While it smoothes out the color transitions, it also introduces transitional tones that might not be intended in the palette.
    * **Dithering** — determines whether an image will be dithered in the process of quantization or not. In simple terms, it means reducing the color banding while using a limited color palette.
* **Pixelization** — rounding to a specific virtual pixel size.
    * **Pixel size** — size of a virtual pixel. For example, at value of `8`, an image with resolution 1024x512 will _appear_ as if its resolution were 128x64—its actual resolution won't be affected.
* **Sharpening** — unsharp masking.
    * **Strength** — sharpening strength.
    * **Radius** — sharpening radius.
* **Symmetry** — makes the frame symmetrical.
    * **Horizontal** — symmetrize the horizontal axis.
    * **Vertical** — symmetrize the vertical axis.
* **Transformation** — geometric transformations applied to the entire frame.
    * **Translation X/Y** — amount of shifting to apply (e.g. X 0.3 — 30% of the image width).
    * **Rotation** — amount of rotation to apply; degrees.
    * **Scaling** — amount of scaling/zooming to apply (e.g. 2.0 — twice as large).


* **Color level mean** — mean value measuring per RGB channel.


* **Color level sigma** — standard deviation measuring per RGB channel.


* **Luminance mean** — luminance mean measuring.


* **Luminance sigma** — luminance standard deviation measuring.


* **Noise sigma** — noise standard deviation measuring.


## Averaging
Averaging of last generated frames.

**NOTE:** Slows down the morphing effect, increases the general middle-scale detail precision, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** filter).

### Frame count
Amount of last generated frames to be blended together to produce a final frame.

### Trimming
Factor of distribution trimming (e.g. `0.25` trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.

**NOTE:** The higher this value is, the longer it will take for any visible changes to occur: factor of `0.5` will require approximately half of **Frame count** first iterations.

### Easing
Frame averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each previous frame; at the value of `0` all frames will be blended evenly.

* **NOTE:** This parameter won't have any effect if **Trimming** is greater than `0`.

### Preference
"Brightness preference" of the averaging algorithm. At minimum, it prefers the darkest colors, at maximum—the brightest ones.

**NOTE:** The greater the **Trimming** is, the less this parameter will affect the result.


* **Interpolation** — interpolation of the current image towards the new image.
    * **Blending** — rate of introduction of colors from the new image.
    * **Movement** — rate of spatial shifting towards the similar areas of the new image.
    * **Radius** — radius of similar area detection.


* **Limiting** — limiting of the difference between the previous and the current image.
    * **Mode** — limiting mode.
        * **Clamp** — clamp the difference, cutting off anything higher than **Maximum difference**.
        * **Compress** — compress the difference, "squashing" its range to **Maximum difference**.
    * **Maximum difference** — maximum difference between the values of the individual color channels.
        * **NOTE:** This value represents both positive and negative values.


* **Random sampling** — random picking of pixels from the new image.
    * **Chance** — chance of pixels from the new image to appear in the current image.


* **Saving** — automatic saving of resulting images.
    * **Save every N-th frame** — stride at which the frames will be saved.
    * **Archive mode** — disable saving of metadata inside of each frame (such as prompt, seed, etc.) and enable maximum compression.


* **Video rendering** — automatic video rendering.
    * **Render draft/final every N-th frame** — stride at which a draft/final video will be rendered.
    * **Render draft/final on finish** — determines if a draft/final video will be rendered after finishing all iterations.
    * **NOTE:** All of the actual video configuration options are located in the **Video Rendering** tab.


* **Detailing** — an additional detailing pass that upscales the image and then scales it back, allowing for much higher precision at the cost of the processing speed.
    * **Scale** — upscaling factor.
        * **NOTE:** It doesn't affect the final output resolution, but rather the processing resolution itself.
    * **Sampling method** — same as the standard img2img option.
    * **Steps** — same as the standard img2img option.
    * **Denoising strength** — same as the standard img2img option.

* **Processing** — the main Stable Diffusion processing procedure.
    * **NOTE:** Currently, all of the settings listed here are related to the averaging of several samples generated from a single frame.
    * **Sample count** — amount of samples to take for generating a frame.
        * **NOTE:** Reduces the jittering between the consecutive frames, increases the general middle-scale detail precision, multiplies amount of work to process each frame correspondingly, and makes the resulting frames blurrier (can be somewhat mitigated by enabling the **Sharpening** preprocessing effect).
    * **Trimming** — factor of distribution trimming (e.g. `0.25` trims 25% of the darkest and brightest colors), controlling the sharpness and bringing means closer to the median.
    * **Easing** — sample averaging easing factor. The more this value is, the sharper is the blending curve, leading to less contribution for each subsequent sample; at the value of `0` all samples will be blended evenly.
        * **NOTE:** This parameter won't have any effect if **Trimming** is greater than `0`.
    * **Preference** — "brightness preference" of the averaging algorithm. At minimum, it prefers the darkest colors, at maximum—the brightest ones.
        * **NOTE:** The greater the **Trimming** is, the less this parameter will affect the result.



# Video Rendering Tab
**NOTE:** Draft mode skips all video filters, making the rendering process much faster for preview purposes.

**NOTE:** Resulting videos will be placed into the `<project subdirectory>/videos` directory.

## Frames per second
Virtual framerate. It corresponds to how often the frames change, but not necessarily to the actual video framerate, which is still a subject to change by the interpolation.

## Looping
Makes the resulting video loop in a "boomerang"-like fashion (e.g. `1 2 3 4 3 2 1`).

## (A bunch of accordions)
See the [Video Filters](#video-filters) section.

**NOTE:** All of these accordions can be reordered manually (using mouse or touchscreen, for example) to determine the order in which the filters will be invoked.

## Parallel index
Index of an image set to render.

## Render draft/final
Start the video rendering immediately.

## Preview
A video player that will show the result after the _manually started_ video rendering finishes.



# Video Filters
**NOTE:** Order matters a lot. For example, applying **Sharpening** after **Text overlay** will make the text sharpened, but otherwise the text will be drawn _on top_ of a sharpened video.

## Generic options

### (✔️ to the left of a filter name)
Determines if a filter is enabled.

## Chromatic aberration
Fake chromatic aberration-like effect that shifts red and blue channels away from the pixel's center.

### Distance
Distance of channel shifting in pixels.

## Color balancing
Common color balancing.

### Brightness
Target brightness level.

### Contrast
Target contrast level.

### Saturation
Target saturation level.

## Deflickering
Reduction of the luminance variations between the consecutive frames.

### Frames
Amount of frames to take into account when calculating the mean luminance.

## Interpolation
Video framerate upscaling/downscaling using motion interpolation in order to keep the transitions between frames smooth.

### Frames per second
Interpolated video framerate.

### Motion blur subframes
Additional subframe count to make the resulting video even smoother.

**NOTE:** Results are mostly negligible, and each subframe multiplies the amount of work by the factor of `x + 1`.

## Scaling
Video resolution upscaling/downscaling using Lanczos interpolation.

### Width/Height
Scaled video resolution.

### Padded
Pad video with borders if the aspect ratio doesn't match, otherwise simply stretch it to fill **Width/Height**.

### Background color
Color of the padded area.

### Backdrop
Use a scaled copy of the video as the background.

### Backdrop brightness
Brightness of the backdrop video.

### Backdrop blurring
Blurring radius of the backdrop video.

## Sharpening
Unsharp masking.

### Strength
Sharpening strength.

### Radius
Sharpening radius.

## Temporal averaging
Averaging of several consecutive frames.

### Radius
Filter radius; total amount of averaged frames equals to `x * 2 + 1`.

### Algorithm
Algorithm to use when computing the average.

* **Mean** — produces blurry video.
* **Median** — produces sharper video than **Mean**, but more prone to artifacts.

### Easing
Kernel easing factor.

**NOTE:** This parameter is relevant only for **Mean** algorithm.

* Value of 0 means that every frame will be averaged in an equal proportion, whereas value greater than 0 makes a distribution ranging from 0 to 1.
* Value greater than 0 makes a soft distribution curve.
* Value of 1 makes a triangle distribution curve.
* Value greater than 1 makes a sharp distribution curve.

Examples:
* `Radius: 1; Easing: 1.0 = Weights [0.5 1 0.5]`
* `Radius: 1; Easing: 0.5 = Weights [0.707 1 0.707]`
* `Radius: 1; Easing: 2.0 = Weights [0.25 1 0.25]`
* `Radius: 3; Easing: 0.0 = Weights [1 1 1 1 1 1 1]`

## Text overlay
Text drawing on top of the video.

### Text
Text that will be drawn. Variables should be enclosed between `{}`. Available variables are:
    * **frame** — number of the currently shown frame.

### Anchor X/Y
Anchor of where the text should be placed regarding the frame borders. 0.0 — left/top, 0.5 — center/center, 1.0 — right/bottom.

### Offset X/Y
Position of the text relative to the anchor, measured in pixels.

### Font
Name of the system-installed font.

### Font size
Size of the font, measured in pixels.

### Text color
Color of the text.

### Shadow offset X/Y
Offset of the text shadow, measured in pixels.

### Shadow color
Color of the text shadow.



# Measuring Tab

## Render plots
Start the plot rendering immediately.

## Plots
Previews of the _manually rendered_ plots.



# Tools Tab
Various utilities for managing the currently selected project.

## Delete intermediate frames
Delete all frames in the project's folder other than the first one and the last one.

## Delete session data
Delete the temporary session data.

**WARNING:** This will make the project unable to be continued—only restarting with the same parameters will be possible.



# Settings Tab

## Output

### Output directory
Main directory to which all of the extension's projects will be saved.

### Autosave every N iterations
Save the project's data automatically over each N full iterations of the pipeline.

## Live preview

### Show only finished images
Set only the fully processed frames as live preview.

### Parallel index
Index of an image from the batch to preview.

**NOTE:** Can be set to `0` for previewing an entire batch.

## Processing

### Pixels per batch
Upper limit of pixels (`width * height`) below which images will be rendered in parallel, potentially speeding up the processing. For example, if this option is set to `1048576` (equals to an image of size `1024x1024`, `2048x512`, and so on), then exactly four images of size `512x512` can be processed in a single batch.

## UI

### Preset sorting order
Determines how presets are sorted inside the preset menu.

### Project sorting order
Determines how projects are sorted inside the project menu.

### Gallery size
Amount of images to show in the project preview gallery.
