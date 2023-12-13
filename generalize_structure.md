# Modular Denoise Latents Node
The Modular Denoise Latents node inherits from the Denoise Latents node, and adds the ability to use multiple denoise methods in parallel or in sequence. It overrides the create_pipeline() function and provides override paths for custom and nested StableDiffusionGeneratorPipeline objects.  

The modular input field of the node is a list of nested dictionaries, each of which contains the following fields:
- `function`: Reference to the function call to use for noise prediction.
- `kwargs`: Dictionary of keyword arguments to pass to the function call.
- `nested call` : None OR another nested dictionary with the same structure. If not None, the function call will be passed the output of the nested call as an additional argument.

## Structure
The Modular Denoise Latents node is structured as follows:
```
ModularDenoiseLatentsNode
├── create_pipeline() # Overrides DenoiseLatentsNode.create_pipeline()

StableDiffusionGeneratorPipeline
├── step() # Overrides StableDiffusionGeneratorPipeline.step()


To add:  
- MultiDiffusion
- Dilated Sampling (add butterworth filtering option?)
- Skip Residual
- Tiled Denoise
- SDXL Color Correction

## Update: That's all not going to work.  
Primarily because function call references are not serializable. Can't pass them between nodes.  
### New Strategy!
Handle imports for functions from a special modifiers folder, and pass in names instead of direct references.