# Modular Denoise Latents
This node pack provides an injection point in a Denoise Latents Node that accepts replacement functions to modify the noise prediction process. This allows rapid development of customized pipelines as well as the ability to combine techniques from multiple research papers without having to edit Denoise Latents every time.  

Please Note: I am not the original designer for most of these noise prediction methods. This is the work of talented and knowledgeable people. I am simply porting their discoveries into an architecture that lets me more easily manipulate and combine them.  
| Node | Usage | Source |
| --- | --- | --- |
| Standard UNet Step Module | Calls up the default noise prediction behavior. Should be the same as not connecting a module input, unless someone forgets to check for that behavior. | InvokeAI Base |
| MultiDiffusion Module | Splits the denoise process into multiple overlapping tiles. Adds generation time but reduces VRAM usage. Randomly shifts tiles each step to prevent visible seams. The random movement requires an additional buffer to be added around the latent. The buffer padding mode can be selected on the node. Breaks with t2i adapters. | https://multidiffusion.github.io/ |
| Dilated Sampling Module | Splits the denoise process into multiple interwoven latent tiles. Reduces VRAM usage. Dramatically reduces quality. Used in the DemoFusion process to maintain structure for MultiDiffusion via a cosine decay transfer. | https://ruoyidu.github.io/demofusion/demofusion.html |
| Cosine Decay Transfer | Smoothly changes over from one pipeline to another based on the remaining denoise value. Higher decay values swap over sooner. | https://ruoyidu.github.io/demofusion/demofusion.html |
| Linear Transfer | Smoothly changes over from one pipeline to another based on the current step index. | N/A |
| Latent Color Guidance | Still considering the best way to go about this. Adjusts denoise process to keep color distribution near average. Fixes yellow drift in SDXL. Boost color range as well. | https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space |
| Tiled Denoise | Splits the denoise process into multiple overlapping tiles. Adds generation time but reduces VRAM usage. Tile positions are maintained with a static minimum overlap. | N/A |
| Skip Residual | Instead of predicting noise, create a timestep% noised version of the input latent. | https://ruoyidu.github.io/demofusion/demofusion.html |
| Analyze Latents | I originally put this in for debugging, but it's pretty helpful for tuning the color guidance module. | N/A |
| CFG Rescale | Scales the predicted noise relative to the unconditional noise prediction sigma. | https://arxiv.org/pdf/2305.08891.pdf |

## Using modules
The new Modular Denoise Latents node added by this pack includes an input for "Custom Modules". The node inherits from InvokeAI's built-in Denoise Latents Invocation, and if you do not connect any modules then it will function exactly the same as the default denoise node does. As a result, these module nodes should also be forward-compatible with almost any updates to InvokeAI. You can use the Modular Denoise Latents node anywhere that you would use the normal node in case you want to add modules in the future without having to remake the rest of the connections.  
Modules can be connected into each other as sub-modules in a tree structure. Transfer modules will change the noise prediction from one pipeline to the other. Normal noise prediction modules will apply their sub-modules in their internal process.

### Example: SDXL Color Correction
The SDXL base model has a bias issue in its latent representation that causes many subjects (most notably people) to appear slightly too yellow. Certain prompts and styles will have different biases, and some finetunes already correct (or overcorrect) for this.  
To lock the colors into a more average distribution, we can use a Color Guidance Module with the channels set to Colors Only and a Mean of 0. I am starting the process at step 10 to ensure that the images do not differ in structure for this comparison (composition can change if the colors are corrected in the initial noise stages). An end step of -1 means the correction will continue until the final step.  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/4c9fdfd0-ed7e-4e8d-9112-cea6509ecb5b)  

The SDXL Color channels are L1 and L2. The brightness and contrast (L0, L3) are largely unchanged, but the coloration is more natural.  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/d7d07ad6-ff97-4e8a-b7ca-a0f526ca2229)  

Note: Color Guidance can be applied to SD1.5 models, but the individual latent channels are not as independently tied to colors as they are in SDXL. Some experimentation is required, and there may be a separate node for SD1.5 in the future if there turns out to be a controllable use case.  

### Example: Tiled Denoise Upscaling
Tiled denoise becomes much more consistent when the tiles reconcile their difference between each step. This results in significantly less visible seams in the output. Splitting the denoise process into tiles decreases the VRAM usage for large upscales. In this setup, a 512x512 image is being scaled to 2048x2048 before being passed into a denoise latents node with a Tiled Denoise module.
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/f2f800bd-1d6f-491f-b2c6-736ca13a6369)  


Original Image (512x512):  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/4ef9eb88-4ac0-4e00-92cf-e2cdbf160547)  

Tiled Denoised (2048x2048):  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/20a8d545-8105-4c71-b821-e98132f53ef2)  
The result is very consistent, even though the denoise start was set to 0.6; the overlap between tiles was able to maintain the composition correctly.

### Transfer Functions
Sometimes in advanced setups you don't want the effects of a pipline to apply for the entire process, or you want to switch from one pipeline to another midway through. For this, there are two Transfer functions that accept multiple sub-module inputs.  

**Linear Transfer:**  
The Linear Transfer Module will smoothly interpolate between two input pipelines based on the current step index. Until the Start Step, the linear transfer will only process the first pipeline. After the End step, it will only process the second pipeline. Between the two steps, both pipelines will be process and combined based on the progress bewteen start and end.  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/be5f95c4-6726-4316-be1e-682c25e702bd)  
Example: If the start and end steps are 10 and 20, respectively, then at step 15 of the denoise process the linear transfer will use 50% of each input. At step 19 it will use 10% of the first submodule and 90% of the second.  

**Cosine Transfer:**   
The Cosine Transfer Module will smoothly shift from one pipeline to another based on the denoise % at each step. The decay value is an exponent: high values will make the pipelines change earlier, low values later.  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/d523a1f4-0d05-4c72-92af-36fdf48113b7)  
Important Note: Neither pipeline ever has a 0% effect when using a cosine transfer. This means that any time you use cosine to interpolate between two module pipelines, you are doubling the amount of processing that needs to happen since each pipline has to denoise independently and then be merged together.  

### Example: Chaining Transfers
In this DemoFusion implementation, a very high resolution image is being passed in as part of an upscale workflow. The beginning of the denoise process is using Skip Residual to prevent it from changing the composition. The Skip Residual uses a Cosine Transfer to change over to a combination of Dilated Sampling (which maintains structure during the middle of the denoise process) and MultiDiffusion (tiled denoise with random tile placement to prevent seams and add detail).  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/d8c532f6-3b97-4182-9b1d-7246de9c9241)  

The total impact of each module ends up like this:  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/f0c49d22-0e6f-404c-94c5-317eb7b015b1)  
Different decay values can impact when each stage takes over.

### Example: Order of Execution
These pipelines are not the same:  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/3c2b1ef4-34bc-466e-b2c8-8af6dc35f019)  
In the first pipeline, the Color Guidance node is a submodule of Tiled Denoise. As a result, color guidance will be applied to each tile individually. In the second pipeline, Tiled Denoise is a submodule of Color Guidance, so the color shift will affect the average for the entire image outside of the tiled denoise process.  
Here you can see the effects of the two. In the end, the total average brightness is the same. However, the first pipeline (left) has distinct banding where the individual tiles were attempting to match a target average brightness instead of the full image brightness being adjusted.  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/0d9bc95a-52bd-4ba6-9f27-284cf2887652)

## Developing New Modules
Coming soon! For now if you want to code something new then just reference the ones I have here. The structure is very repetitive and simple to implement.

## Planned/Prospective Changes:  
| Feature | Type | Usage |
| --- | --- | --- |
| ScaleCrafter | module | Original implementation of dilated sampling to get high resolution results. Not sure if it is applicable here, will need to look deeper into it. |
| Perp negative | module | A more precise application of negative conditioning during noise prediction |
| Adversarial Inpainting | module | Theoretically adds inpaint objects but replaces result with original where not significantly changed. Note to self: Noise Prediction = lerp(P, S, -wD) |
| Architecture Tutorial | documentation | Need to create explanations for extending with custom modules. |
