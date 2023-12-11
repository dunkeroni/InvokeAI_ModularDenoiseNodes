# InvokeAI_DemoFusion
InvokeAI Nodes implementation of the DemoFusion paper.  
Original source: https://ruoyidu.github.io/demofusion/demofusion.html  

## Planning  
1. Integrate with (or inherit from) existing Denoise Latents node. (2/3 DONE)  
   - Multidiffusion:  
     - Implemented.  
     - Window size is the larger of the two downscaled dimensions.  
     - Default buffer strategy causes fringing on edges. Might experiment with improving that.  
   - Dilated sampling:  
     - Implemented.  
     - Downsampling scale must be a whole number divisor of width and heigh, who's factor must then still be divisible by 8.  
     - Can be used without multidiffusion, but results are bad.  
   - Skip Residual:  
     - Not Yet Implemented.  
     - Might be architecturally weird to add it with the current implementation.  
     - Not very necessary unless using high denoise, which is a counterproductive use case IMO.  
2. Fix Compatibility Breaks (NOT YET DONE)  
   - ControlNet  
   - IP Adapater  
   - T2I  
3. Error Catching (NOT YET DONE)  
   - Input size incompatible with downsample scale  
   - Input limiting on decay ratios  
4. Abstract an generalize for future noise prediction modifiers techniques? (perp-neg, etc.)  

