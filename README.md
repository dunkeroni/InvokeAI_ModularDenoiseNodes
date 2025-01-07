# Modular Denoise Latents
This node pack originally provided injection points in a Denoise Latents Node in order to modify the noise prediction process. This functionality has since been built into the core Invoke nodes, but it is not exposed to users without setting an environment variable. To get around this, I provide an Exposed Denoise Latents node with an extra input and handling for extensions.

Please Note: I am not the original designer for most of these noise prediction methods. This is the work of talented and knowledgeable people. I am simply porting their discoveries into an architecture that lets me more easily manipulate and experiment with them.  

IMPORTANT: Many things will be frequently broken in this repo. I will try to keep old code archived somewhere so it can be more easily updated and reintroduced to new versions, but I frequently scrap everything that works and build new things.  