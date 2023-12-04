# InvokeAI_DemoFusion
InvokeAI Nodes implementation of the DemoFusion paper

## Planning  
1. ~~Runable Example code from nodes interface~~ clunky and not able to integrate well  
   - Status: partially implemented  
      - Requires Euler sampler to work without error.  
      - Produces terrible results.  
2. ~~Refactor into Upscaler (pass in pre-generated latents)~~  upscale on your own time.  
3. Integrate with (or inherit from) existing Denoise Latents node.  
   - Status: Current implementation.
      - Works in a somewhat stable fashion.
      - Have not tested control inputs yet. Probably they will break.  