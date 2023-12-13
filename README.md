## Repository Changing:
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
| Cosine Decay Transfer | Smoothly changes over from one pipeline override to another. Higher decay values swap over sooner. |
| Linear Transfer | NOT YET IMPLEMENTED |
| Skip Residual Module | NOT YET IMPLEMENTED |
| Latent Color Correction | NOT YET IMPLEMENTED |
| Tiled Denoise | NOT YET IMPLEMENTED |

