# InvokeAI_DemoFusion
InvokeAI Nodes implementation of the DemoFusion paper

## Planning  
1. Runable Example code from nodes interface  
   - Status: partially implemented  
      - Requires Euler sampler to work without error.  
      - Produces terrible results.  
2. Refactor into Upscaler (pass in pre-generated latents)
3. Integrate with (or inherit from) existing Denoise Latents node.  