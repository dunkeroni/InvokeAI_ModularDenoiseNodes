# Modular Denoise Latents
This node pack provides an injection point in a Denoise Latents Node that accepts replacement functions to modify the noise prediction process. This allows rapid development of customized pipelines as well as the ability to combine techniques from multiple research papers without having to edit Denoise Latents every time.  
  
The "Modular Denoise Latents" node has a "Custom Modules" input that connects from the modular nodes. All modules can be found by searching "modular" in the workflow interface.  
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

Modules can be connected into each other as sub-modules in a tree structure. Transfer modules will change the noise prediction from one pipeline to the other. Normal noise prediction modules will apply their sub-modules in their internal process. Example: MultiDiffusion will split the latent into tiles, and then use its sub-module pipeline to process each tile individually.
![image](https://github.com/dunkeroni/InvokeAI_DemoFusion/assets/3298737/06fc0004-830b-4895-bf0e-b97976b612b1)  

Update 10-15-2023: Fixed ControlNet inputs to work with Multidiffusion/Tile sampling. Dilated sampling has been changed to drop controlnet inputs.  
Update 10-18-2023: Fixed t2i adapter inputs to work with Tile sampling. Causes bad smearing on multidiffusion, but doesn't break. Might add a toggle options to keep/drop it.  

## Planned/Prospective Changes:  
| Feature | Type | Usage |
| --- | --- | --- |
| ScaleCrafter | module | Original implementation of dilated sampling to get high resolution results. Not sure if it is applicable here, will need to look deeper into it. |
| Perp negative | module | A more precise application of negative conditioning during noise prediction |
| Adversarial Inpainting | module | Theoretically adds inpaint objects but replaces result with original where not significantly changed. Note to self: Noise Prediction = lerp(P, S, -wD) |
| Architecture Tutorial | documentation | Need to create explanations for extending with custom modules. |
