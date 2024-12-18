##########################################################################################################################
# From: https://arxiv.org/pdf/2411.18552v1
# Title: FAM Diffusion: Frequency and Attention Modulation for High-Resolution Image Generation with Stable Diffusion
##########################################################################################################################

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.invocations.fields import (
    InputField,
    LatentsField,
)

import torch
from .extension_classes import GuidanceField, base_guidance_extension, GuidanceDataOutput
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.util.logging import info, warning, error
import random
import einops
from diffusers import UNet2DConditionModel
from typing import Type, Any
from .attention_modulation import StoreAttentionModulation
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode


@base_guidance_extension("FAM_FM")
class FAM_FM_Guidance(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
        c: float,
        latent_image_name: str,
    ):
        self.c = c
        self.initial_latents = context.tensors.load(latent_image_name)
        self.noise = torch.randn(
            self.initial_latents.shape,
            dtype=torch.float32,
            device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(random.randint(0, 2 ** 32 - 1)),
        ).to(device=self.initial_latents.device, dtype=self.initial_latents.dtype)
        super().__init__()
    
    @callback(ExtensionCallbackType.PRE_STEP)
    @torch.no_grad()
    def pre_step(self, ctx: DenoiseContext):
        t = ctx.timestep
        if t.dim() == 0:
            t = einops.repeat(t, "-> batch", batch=ctx.latents.size(0))
        
        latents = ctx.latents.clone().double()
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents, s=None, dim=(-2, -1), norm="ortho"))
        skip_residual = ctx.scheduler.add_noise(self.initial_latents, self.noise.to(self.initial_latents.device), t).double()
        skip_residual_fft = torch.fft.fftshift(torch.fft.fft2(skip_residual, s=None, dim=(-2, -1), norm="ortho").to(ctx.latents.device))
        K_t = torch.ones_like(self.initial_latents).to(ctx.latents.device)
        
        rho = ctx.timestep.item() / ctx.scheduler.config.num_train_timesteps
        h_i = self.initial_latents.shape[-2]
        w_i = self.initial_latents.shape[-1]
        tau_h = h_i * self.c * (1 - rho)
        tau_w = w_i * self.c * (1 - rho)

        h_d = latents.shape[-2]
        w_d = latents.shape[-1]

        # create a high-pass filter on the shifted domain
        # in horizontal dimension: K_t = rho if |X - Xc| < tau_w/2 else 1
        # in vertical dimension: K_t = rho if |Y - Yc| < tau_h/2 else 1
        K_t[:, :, int((h_i // 2) - (tau_h // 2)): int((h_i // 2) + (tau_h // 2)), int((w_i // 2) - (tau_w // 2)): int((w_i // 2) + (tau_w // 2))] = 1 - rho #paper formulas are wrong, missing 1-

        lf_part = skip_residual_fft * (1 - K_t)

        # pad the low frequency part equally in both directions with zeros
        pad_h = h_d - h_i
        pad_w = w_d - w_i
        lf_part = torch.nn.functional.pad(lf_part, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0)
        K_t_padded = torch.nn.functional.pad(K_t, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=1)

        #combine the low frequency components of the skip residual with the high frequency components of the latent image
        latents_fft = latents_fft * K_t_padded + lf_part

        #invert the FFT to get the new latent image
        ctx.latents = torch.fft.ifft2(torch.fft.ifftshift(latents_fft), s=None, dim=(-2, -1), norm="ortho").real.half()


@invocation(
    "frequency_modulation_extInvocation",
    title="I2I Preservation (FM) [Extension]",
    tags=["FAM", "frequency", "modulation", "extension"],
    category="latents",
    version="1.0.3",
)
class FAM_FM_ExtensionInvocation(BaseInvocation):
    """Preserves low frequency features from an input image."""
    c: float = InputField(
        title="c",
        description="'c' value for the FAM extension. Affects scaling of the cutoff frequency per step.",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    latent_image: LatentsField = InputField(
        title="Latent Image",
        description="Latent image to be targeted.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {
            "c": self.c,
            "latent_image_name": self.latent_image.latents_name,
        }
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="FAM_FM",
                extension_kwargs=kwargs
            )
        )



def patch_unet_attention_processor(unet: UNet2DConditionModel, processor_cls: Type[Any]):
    """A context manager that patches `unet` with the provided attention processor.

    Args:
        unet (UNet2DConditionModel): The UNet model to patch.
        processor (Type[Any]): Class which will be initialized for each key and passed to set_attn_processor(...).
    """
    unet_orig_processors = unet.attn_processors

    # create separate instance for each attention, to be able modify each attention separately
    unet_new_processors = {key: processor_cls() for key in unet_orig_processors.keys()}
    try:
        unet.set_attn_processor(unet_new_processors)
        yield None

    finally:
        unet.set_attn_processor(unet_orig_processors)


@base_guidance_extension("FAM_AM")
class FAM_AM_Guidance(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
        l: float,
        latent_image_name: str,
    ):
        self.l = l
        self.initial_latents = context.tensors.load(latent_image_name)
        self.noise = torch.randn(
            self.initial_latents.shape,
            dtype=torch.float32,
            device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(random.randint(0, 2 ** 32 - 1)),
        ).to(device=self.initial_latents.device, dtype=self.initial_latents.dtype)
        self.dummy_manager = ExtensionsManager()
        super().__init__()

    def is_custom_attention(self, key) -> bool:
        """ IMPORTANT:
            The custom attention is SLOW and FAT.
            Setting all processors to use the custom attention makes the process take 2x longer
            It also requires an extra 12GB of GPU memory at SDXL 1024x1024 resolution just to hold coppies of the attention weights.
            The paper specifies that they only use it for the up_blocks, and that the most significant effect is up_block_0.
            There 140 processors in the SDXL unet, with 20 in up_blocks.0 and 12 in up_blocks.1 (32 total).
            A huge ammount of the VRAM and most of the time disparity is coming from up_blocks.1 (hidden states shape [2, 10, 4096, 64] vs [2, 20, 1024, 64])
        """
        #key is in form 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.processor'
        blocks = key.split('.')
        if blocks[0] == 'up_blocks':
            if (blocks[1] == "0" and blocks[3] == "2"):# or blocks[1] == "1":
                print(f"Custom attention for {key}")
                return True


    

    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def pre_denoise_loop(self, ctx: DenoiseContext):
        unet_replacement_processors = {}
        self.unet_new_processors = []

        for key in ctx.unet.attn_processors.keys():
            if self.is_custom_attention(key):
                unet_replacement_processors[key] = StoreAttentionModulation(self.l)
                self.unet_new_processors.append(unet_replacement_processors[key])
            else:
                unet_replacement_processors[key] = ctx.unet.attn_processors[key]

        ctx.unet.set_attn_processor(unet_replacement_processors)


    
    @callback(ExtensionCallbackType.PRE_STEP)
    @torch.no_grad()
    def pre_step(self, ctx: DenoiseContext):
        t = ctx.timestep
        if t.dim() == 0:
            t = einops.repeat(t, "-> batch", batch=ctx.latents.size(0))
        self.stored_latents = ctx.latents.clone()
        ctx.latents = ctx.scheduler.add_noise(self.initial_latents.to(ctx.latents.device), self.noise.to(ctx.latents.device), t)
        ctx.latent_model_input = ctx.scheduler.scale_model_input(ctx.latents, ctx.timestep)

        #set all the processors to store the attention weights
        for attn_processor in self.unet_new_processors:
            attn_processor.store_copy = True
        
        #call the unet step to get the attention weights
        ctx.sd_backend.run_unet(ctx, self.dummy_manager, ConditioningMode.Both)

        #Change back to false, attentions will use the stored maps in the real unet pass
        for attn_processor in self.unet_new_processors:
            attn_processor.store_copy = False
        
        ctx.latents = self.stored_latents
        

@invocation(
    "attention_modulation_extInvocation",
    title="I2I Preservation (AM) [Extension]",
    tags=["FAM", "attention", "modulation", "extension"],
    category="latents",
    version="1.0.0",
)
class FAM_AM_ExtensionInvocation(BaseInvocation):
    """Preserves low frequency features from an input image."""
    l: float = InputField(
        title="l",
        description="'c' value for the FAM extension. Affects scaling of the cutoff frequency per step.",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    latent_image: LatentsField = InputField(
        title="Latent Image",
        description="Latent image to be targeted.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {
            "l": self.l,
            "latent_image_name": self.latent_image.latents_name,
        }
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="FAM_AM",
                extension_kwargs=kwargs
            )
        )