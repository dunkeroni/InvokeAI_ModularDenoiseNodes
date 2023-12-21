Previously this repository was an implementation of DemoFusion for InvokeAI. It has now been generalized to allow more custom pipeline modifications.  
DemoFusion paper: https://ruoyidu.github.io/demofusion/demofusion.html  

## Code Refactor:
There were some limitations to the way modules worked. Some structure changes to what happens when and which parameters are passed has improved that.  
The new structure is as follows:  
- Original Latents (NOT scaled model inputs), ControlNet DATA, T2I DATA are all passed down as inputs to the modules.  
- Lowest level modules are responsible for resolving Latents->Scaled Latents as well as CNet/T2I->Residuals  
- Modules return a Tuple of tensors: Noise_Prediction and Original_Latents are passed UP as return values to the calling module.  
Modules have the opportunity to modify original latents (previous step result) before or after they are used for noise prediction. This is necessary for color correction and skip residual.  

# Modular Denoise Latents
The "Modular Denoise Latents" node has a "Custom Modules" input that accepts inputs from the modular nodes. These modules override the default behavior of the noise prediction in the diffusion pipeline.  
All modules can be found by searching "modular" in the workflow interface.  
| Node | Usage | Source Link |
| --- | --- | --- |
| Standard UNet Step Module | Calls up the default noise prediction behavior. Should be the same as not connecting a module input, unless someone forgets to check for that behavior. | InvokeAI Base |
| MultiDiffusion Module | Splits the denoise process into multiple overlapping tiles. Adds generation time but reduces VRAM usage. Randomly shifts tiles each step to prevent visible seams. The random movement requires an additional buffer to be added around the latent. The buffer padding mode can be selected on the node. Breaks with t2i adapters. | https://multidiffusion.github.io/ |
| Dilated Sampling Module | Splits the denoise process into multiple interwoven latent tiles. Reduces VRAM usage. Dramatically reduces quality. Used in the DemoFusion process to maintain structure for MultiDiffusion via a cosine decay transfer. | https://ruoyidu.github.io/demofusion/demofusion.html |
| Cosine Decay Transfer | Smoothly changes over from one pipeline to another based on the remaining denoise value. Higher decay values swap over sooner. | https://ruoyidu.github.io/demofusion/demofusion.html |
| Linear Transfer | Smoothly changes over from one pipeline to another based on the current step index. | N/A |
| Latent Color Guidance | CURRENTLY BROKEN. Still considering the best way to go about this. Adjusts denoise process to keep color distribution near average. Fixes yellow drift in SDXL. Boost color range as well. | https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space |
| Tiled Denoise | Splits the denoise process into multiple overlapping tiles. Adds generation time but reduces VRAM usage. Tile positions are maintained with a static minimum overlap. | N/A |
| Skip Residual | Instead of predicting noise, create a timestep% noised version of the input latent. | https://ruoyidu.github.io/demofusion/demofusion.html |

Modules can be connected into each other as sub-modules in a tree structure. Transfer modules will change the noise prediction from one pipeline to the other. Normal noise prediction modules will apply their sub-modules in their internal process. Example: MultiDiffusion will split the latent into tiles, and then use its sub-module pipeline to process each tile individually.
![image](https://github.com/dunkeroni/InvokeAI_DemoFusion/assets/3298737/06fc0004-830b-4895-bf0e-b97976b612b1)  

Update 10-15-2023: Fixed ControlNet inputs to work with Multidiffusion/Tile sampling. Dilated sampling has been changed to drop controlnet inputs.  
Update 10-18-2023: Fixed t2i adapter inputs to work with Tile sampling. Causes bad smearing on multidiffusion, but doesn't break. Might add a toggle options to keep/drop it.  

## Planned/Prospective Changes:  
| Feature | Type | Usage |
| --- | --- | --- |
| ScaleCrafter | module | Original implementation of dilated sampling to get high resolution results. Not sure if it is applicable here, will need to look deeper into it. |
| Adversarial Inpainting | module | Theoretically adds inpaint objects but replaces result with original where not significantly changed. Note to self: Noise Prediction = lerp(P, S, -wD) |
| T2I adapter compatibility | module | Currently broken for modules that split up the latent for submodules |
| Add versioning and changelog | Documenation | Will make a "1.0.0" release of this repository once things are less broken |
| SDGP Persistent Variables | Utility | Some unet modifications require extra info (input latent, etc.) that should be stored in the calling object to avoid loading from context database. Will set up some set/gets. |
