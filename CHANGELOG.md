## Version 1.0.2 - 10/27/2023
Updated compatibility to Invoke 3.5.0
Removed CFG Rescale Module since that is in the default denoise latents node now.

## Version 1.0.1 - 10/26/2023
CFG Rescale Module added

## Version 1.0.0 - 10/24/2023 - First Release 
Modular Denoise Latents is now mostly stable, at least enough to say I won't totally break it every other day.  
Further features and changes will be done in development branches, so main branch should remain usable.  

Features:  
- Modules
    - Base Modules
        - Standard UNet Step (in case you want to be explicit about what is being used)
    - Modifier Modules
        - Tiled Denoise (splits up the denoise process to save on VRAM)
        - MultiDiffusion (Tiled Denoise with random jitter to avoid seams)
        - Dilated Sampling (Part of Demofusion, splits up denoise to interlaced tiles)
        - Color Guidance (SDXL only for now, color correction and shifting during generation)
        - Skip Residual (Part of Demofusion, noises the input image instead of denoising the current latent)
- Bonus: Analyze Latents node, helpful in understanding the results of color guidance.
- Module Capabilities/Scope:
    - CNet/T2i: Modifier Modules can modify/crop the prepared latent data of the adapters
    - Latents: Modules have control over the incoming previous latent, noise prediction latent, and returned "previous" latent.
    - Persistent Data: The Modular_StableDiffusionGeneratorPipeline object retains a dictionary where modules in the stack can store and retrieve information between steps.

