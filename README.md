Previously this repository was an implementation of DemoFusion for InvokeAI. It has now been generalized to allow more custom pipeline modifications.  
DemoFusion paper: https://ruoyidu.github.io/demofusion/demofusion.html  

## Scope Refactor:
There are some limitations to the way these modules work right now. Currently each module passes down the residuals of CNet and T2I, but then recrops and recalculates the data inputs for tile denoising pipelines which adds redundant steps. Additionally the implementation is acting on the noise prediction stage (estimate t+1) and not the step result (estimate t-1). That works for most things so far, but has a reverse effect on color correction/guidance and does not allow for skip residual.  

The Fix: Shift the scope of the module replacement up one layer. Modules will pass the CNet/T2I Data which will be resolved late after all crops and modifications. Modules will return (t-1) final step output instead of noise prediction. Any processes that need to make further insertions to the pipeline will need to account for their adjustments and hand back a corrected (t-1).  

Modules will not interact with or affect IP-Adapter data for the time being. It could be argued that cropping IP Adapter inputs would be helpful, but I think doing so would have detrimental and unpredictable effects on the outputs from a user perspective.

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
| Skip Residual | UNDER CONSTRUCTION. Instead of predicting noise, create a timestep% noised version of the input latent. Doesn't work with the current noise prediction architecture. | N/A | 

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
