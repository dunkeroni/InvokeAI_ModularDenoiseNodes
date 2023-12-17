Previously this repository was an implementation of DemoFusion for InvokeAI. It has now been generalized to allow more custom pipeline modifications.  
DemoFusion paper: https://ruoyidu.github.io/demofusion/demofusion.html  

# Modular Denoise Latents
The "Modular Denoise Latents" node has a "Custom Modules" input that accepts inputs from the modular nodes. These modules override the default behavior of the noise prediction in the diffusion pipeline.  
All modules can be found by searching "modular" in the workflow interface.  
| Node | Usage |
| --- | --- |
| Standard UNet Step Module | Calls up the default noise prediction behavior. Should be the same as not connecting a module input, unless someone forgets to check for that behavior. |
| MultiDiffusion Module | Splits the denoise process into multiple overlapping tiles. Adds generation time but reduces VRAM usage. Randomly shifts tiles each step to prevent visible seams. The random movement requires an additional buffer to be added around the latent. The buffer padding mode can be selected on the node. |
| Dilated Sampling Module | Splits the denoise process into multiple interwoven latent tiles. Reduces VRAM usage. Dramatically reduces quality. Used in the DemoFusion process to maintain structure for MultiDiffusion via a cosine decay transfer. |
| Cosine Decay Transfer | Smoothly changes over from one pipeline to another based on the remaining denoise value. Higher decay values swap over sooner. |
| Linear Transfer | Smoothly changes over from one pipeline to another based on the current step index. |
| Latent Color Guidance | CURRENTLY BROKEN. Fixes denoise process to keep color distribution near average. Fixes yellow drift in SDXL. Boost color range as well. |
| Tiled Denoise | Splits the denoise process into multiple overlapping tiles. Adds generation time but reduces VRAM usage. Tile positions are maintained with a static minimum overlap. |

Modules can be connected into each other as sub-modules in a tree structure. Transfer modules will change the noise prediction from one pipeline to the other. Normal noise prediction modules will apply their sub-modules in their internal process. Example: MultiDiffusion will split the latent into tiles, and then use its sub-module pipeline to process each tile individually.
![image](https://github.com/dunkeroni/InvokeAI_DemoFusion/assets/3298737/06fc0004-830b-4895-bf0e-b97976b612b1)  

Update 10-15-2023: Fixed ControlNet inputs to work with Multidiffusion/Tile sampling. Dilated sampling has been changed to drop controlnet inputs.

## Planned/Prospective Changes:  
| Node | Type | Usage |
| --- | --- |
| ScaleCrafter | module |A specialized implementation of dilated sampling to get high resolution results |
| Skip Residual | module | instead of predicting noise, create a timestep% noised version of the input latent |
| Adversarial Inpainting | module | Theoretically adds inpaint objects but replaces result with original where not significantly changed. Note to self: Noise Prediction = lerp(P, S, -wD) |
| T2I adapter compatibility | module | Currently broken for modules that split up the latent for submodules |
| Add versioning and changelog | Documenation | Will make a "1.0.0" release of this repository once things are less broken |
| SDGP Persistent Variables | Utility | Some unet modifications require extra info (input latent, etc.) that should be stored in the calling object to avoid loading from context database. Will set up some set/gets. |