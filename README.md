# Modular Denoise Latents
This node pack provides injection points in a Denoise Latents Node that modify the noise prediction process. This allows rapid development of customized pipelines as well as the ability to combine techniques from multiple research papers without having to edit Denoise Latents every time.  

Please Note: I am not the original designer for most of these noise prediction methods. This is the work of talented and knowledgeable people. I am simply porting their discoveries into an architecture that lets me more easily manipulate and experiment with them.  

## Version 2.0 and a new API!  
As adaptable as the original implementation was, things were getting a little hairy with the multiple types of module and chains of submodules that made it difficult for users to interact with.  
Version 2 is a complete rewrite with a more thought-out structure that allows more complex interactions and a better development process. The majority of extensions no longer have to be structured as pieces of an abstract recursive stack.  
#### Object-Oriented API:  
The denoise_latents_extensions.py contains a base class called DenoiseExtensionSD12X that all (current) extensions inherit from. This base class provides a standardized structure so that extensions can exist with their own memory and be called by an ExtensionHandlerSD12X at multiple points in the pipeline.
- Each extension provides a list of points where it modifies data as well as any functions that it replaces.
- An unlimited number of extensions can modify at the same point, and they will be ordered by a Priority value.
- Default functions can only be replaced by a single extension at a time, and the handler will call out an error if there are multiple competing extensions at once.
- Modifications are done in-place on a standardized dataclass (with a .copy() function for record keeping) so that function calls don't need a dozen parameters any more.
#### New Modular Denoise Latents Node:  
Previously the denoise latents node itself was inherited from the version built into Invoke. This helped preserve compatibility, but it also forced it to include all of the original inputs in the node interface. The new node is a *copy* of the built-in Denoise Latents, but not a sub-class. This means a few things:  
- Denoise Mask and CFG Rescale inputs have been removed and are relegated to their own guidance extensions.
- A new input to replace the mask input called Additional Guidance has been added. It is a list input for any extensions being used on that denoise latents.
- All of the denoise mask code has been stripped out and condensed into a single extension class location with multiple data modification points.
#### New Capabilities:
- Model Patching functions (LoRA, DoRA, FreeU, HiDiffusion, etc.)
- with() enter and exit handling (mostly for the above functions, but also if your extension has memory that needs to be freed up)
- More injection points for more fine control (start of denoise, before/after scaling, pre-callback, after denoising, etc.)
#### Not Yet Implemented:
- ExtensionHandler interactions for combining multiple lists of extensions (for when swap functions need to apply a different set of extensions to their sub-process)
- Combination/Transfer nodes that interpolate multiple other extensions (a bit tricky since extensions might not trigger at the same point, at the very least requires the above feature to handle sub-stacks)
- Most of the extensions that dealt with conditionings (Invoke's API for those is different now, and I need to go fix them)
- Not all functions are swappable yet. Some further cleanup and modification of the denoise latents node and pipeline are needed first.

## Nodes and Extensions:
#### Currently Available:
| Node | Source | Usage |
| --- | --- | --- |
| Inpaint Mask Guidance | Invoke | Applies inpaint masking to the new API structure. Supports standard and gradient masking, but only the gradient masking node is implemented right now on the user side. |
| Tiled Denoise | https://multidiffusion.github.io/ | Splits the denoise process into multiple overlapping tiles. Adds generation time but reduces VRAM usage. [Optionally] randomly shifts tiles each step to prevent visible seams. The random movement requires an additional buffer to be added around the latent. The buffer padding mode can be selected on the node. Breaks with t2i adapters and ControlNet (for now). |
| Color Guidance SDXL |  https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space | Adjusts denoise process to keep color distribution near average. Fixes yellow drift in SDXL, or lets you specify color/brightness/contrast adjustments. |
| Color Offset SD1 |  https://github.com/Haoming02/sd-webui-vectorscope-cc | Applies offsets to the latent layers between each step to achieved specific RGB color changes. Highly dependent on sampler. Recommend starting with Euler A to test. |
| Sigma Scaling | N/A | Accepts scaling values to construct a piecewise multiplication function that affects sigmas through different parts of the denoise process. Affects detaillevels and sharpness of the result. |
| CFG Rescale | https://arxiv.org/pdf/2305.08891.pdf | Fixes CFG result for ZTSNR models and also helps with too-high CFG values. Used to be an input on the Denoise Latents node. If you don't understand what it's for, then you probably don't need it. |

#### Removed or Under Construction:
| Node | Status |
| --- | --- |
| Override Conditioning | Conditionings are different now, but this will probably wait until multi-pipeline extension handling is supported, since it is most helpful in those use cases. |
| Skip Residual | Originally included for DemoFusion. Might be added back in after multi-pipeline support, but it is not high priority. |
| Dilated Sampling | Ditto above. |
| Gradient Denoise | Original experiment was successfully rolled into Invoke. Use the mask guidance extension for this. |
| Transfer Functions | Coming soon |
| Perp Negative | Need to look into more efficient implementations of this further down in the unet. Also might be cool to use it for regional guidance. |
| Regional Guidance | Exists in a more elegant form in standard Invoke now. |
| Color Gravitation | It was hard to control and the results were never great. Style-Only IP Adapters are a more reliable way to set color palettes. Might revisit if I can find a cleaner implementation. |
| Multidiffusion | Has been combined into Tiled Denoise. They are the same code, but MD has random jitter in the tile placements. This is now a boolean option. |


## Examples:
The following examples are using the old nodes from version 1, but their effects and UI are the same. I will update the images later on when I get around to adding examples for the other new nodes.  
### SDXL Color Correction
The SDXL base model has a bias issue in its latent representation that causes many subjects (most notably people) to appear slightly too yellow. Certain prompts and styles will have different biases, and some finetunes already correct (or overcorrect) for this.  
To lock the colors into a more average distribution, we can use a Color Guidance Module with the channels set to Colors Only and a Mean of 0. I am starting the process at step 10 to ensure that the images do not differ in structure for this comparison (composition can change if the colors are corrected in the initial noise stages). An end step of -1 means the correction will continue until the final step.  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/4c9fdfd0-ed7e-4e8d-9112-cea6509ecb5b)  

The SDXL Color channels are L1 and L2. The brightness and contrast (L0, L3) are largely unchanged, but the coloration is more natural.  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/d7d07ad6-ff97-4e8a-b7ca-a0f526ca2229)  

Note: Color Guidance can be applied to SD1.5 models, but the individual latent channels are not as independently tied to colors as they are in SDXL. Some experimentation is required, and there may be a separate node for SD1.5 in the future if there turns out to be a controllable use case.  

### Tiled Denoise Upscaling
Tiled denoise becomes much more consistent when the tiles reconcile their difference between each step. This results in significantly less visible seams in the output. Splitting the denoise process into tiles decreases the VRAM usage for large upscales. In this setup, a 512x512 image is being scaled to 2048x2048 before being passed into a denoise latents node with a Tiled Denoise module.
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/f2f800bd-1d6f-491f-b2c6-736ca13a6369)  

Original Image (512x512):  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/4ef9eb88-4ac0-4e00-92cf-e2cdbf160547)  

Tiled Denoised (2048x2048):  
![image](https://github.com/dunkeroni/InvokeAI_ModularDenoiseNodes/assets/3298737/20a8d545-8105-4c71-b821-e98132f53ef2)  
The result is very consistent, even though the denoise start was set to 0.6; the overlap between tiles was able to maintain the composition correctly.

