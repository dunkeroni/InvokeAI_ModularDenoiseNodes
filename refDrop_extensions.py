##########################################################################################################################
# From: https://arxiv.org/abs/2405.17661
# Title: RefDrop: Controllable Consistency in Image or Video Generation via Reference Feature Guidance
##########################################################################################################################

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.invocations.fields import (
    InputField,
    LatentsField,
)

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    IPAdapterConditioningInfo,
    IPAdapterData,
    Range,
    SDXLConditioningInfo,
    TextConditioningData,
    TextConditioningRegions,
)
from invokeai.app.invocations.fields import (
    ConditioningField,
    Input,
)
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation

import torch
from .extension_classes import GuidanceField, base_guidance_extension, GuidanceDataOutput
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.util.logging import info, warning, error
import random
import einops
from diffusers import UNet2DConditionModel
from typing import Type, Any, Dict, Iterator, List, Optional, Tuple, Union
from .refDrop_attention import StoreAttentionModulation
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode


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


@base_guidance_extension("RefDrop")
class RefDrop_Guidance(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
        C: float,
        latent_image_name: str,
        skip_up_block_1: bool,
        skip_until: float,
        positive_conditioning: Union[ConditioningField, list[ConditioningField]],
        negative_conditioning: Union[ConditioningField, list[ConditioningField]],
        stop_at: float,
        once_and_only_once: bool
    ):
        self.C = C
        self.initial_latents = context.tensors.load(latent_image_name)
        self.skip_up_block_1 = skip_up_block_1
        self.skip_until = skip_until
        self.positive_conditioning = positive_conditioning
        self.negative_conditioning = negative_conditioning
        self.stop_at = stop_at
        self.once_and_only_once = once_and_only_once
        # self.noise = torch.randn(
        #     self.initial_latents.shape,
        #     dtype=torch.float32,
        #     device="cpu",
        #     generator=torch.Generator(device="cpu").manual_seed(random.randint(0, 2 ** 32 - 1)),
        # ).to(device=self.initial_latents.device, dtype=self.initial_latents.dtype)
        self.dummy_manager = ExtensionsManager()
        self.and_never_again = False
        self.context = context
        super().__init__()

    def is_custom_attention(self, key) -> bool:
        """ IMPORTANT:
            The custom attention is SLOW and FAT.
            Setting all processors to use the custom attention makes the process take 2x longer
            It also requires an extra 12GB of GPU memory at SDXL 1024x1024 resolution just to hold coppies of the attention weights.
            The paper specifies that they only use it for the up_blocks, and that the most significant effect is up_block_0.
            Since there is no published code, it's worth playing around with which ones are activated. Potentially allow a list of string inputs to specify block options.
        """
        #key is in form 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.processor'
        blocks = key.split('.')
        #print(key)
        #if '.attn2.processor' in key:
        #print(f"Custom attention for {key}")
        #if not self.skip_up_block_1 or not 'up_blocks.0.attentions.0' in key: #skip the first up block
        return True


    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def pre_denoise_loop(self, ctx: DenoiseContext):

        self.noise = ctx.inputs.noise.clone()

        unet_replacement_processors = {}
        self.unet_new_processors = []

        for key in ctx.unet.attn_processors.keys():
            if self.is_custom_attention(key):
                unet_replacement_processors[key] = StoreAttentionModulation(self.C)
                self.unet_new_processors.append(unet_replacement_processors[key])
                unet_replacement_processors[key].attn_name = key
                #print(f"added custom attention for {key}")
            else:
                unet_replacement_processors[key] = ctx.unet.attn_processors[key]

        ctx.unet.set_attn_processor(unet_replacement_processors)

        if self.positive_conditioning is None or self.negative_conditioning is None:
            info("At least one of the conditioning fields is None. Using the conditioning data from the context instead.")
            self.ref_conditioning = ctx.inputs.conditioning_data
        else:
            self.ref_conditioning: TextConditioningData = DenoiseLatentsInvocation.get_conditioning_data(
                context = self.context,
                positive_conditioning_field=self.positive_conditioning,
                negative_conditioning_field=self.negative_conditioning,
                cfg_scale=ctx.inputs.conditioning_data.guidance_scale,
                steps = 1 if not isinstance(ctx.inputs.conditioning_data.guidance_scale, list) else len(ctx.inputs.conditioning_data.guidance_scale),
                latent_height=ctx.latents.shape[-2],
                latent_width=ctx.latents.shape[-1],
                device=ctx.latents.device,
                dtype=ctx.latents.dtype,
                cfg_rescale_multiplier=0,
            )

    
    @callback(ExtensionCallbackType.PRE_STEP)
    @torch.no_grad()
    def pre_step(self, ctx: DenoiseContext):
        if self.and_never_again:
            return
        
        t_orig = ctx.timestep
        if self.once_and_only_once:
            ctx.timestep = torch.Tensor([(1 - self.stop_at) * ctx.scheduler.config.num_train_timesteps]).to(ctx.timestep.device, dtype=ctx.timestep.dtype)
        t = ctx.timestep
        if t.dim() == 0:
            t = einops.repeat(t, "-> batch", batch=ctx.latents.size(0))
        timestep_fraction = 1 - (ctx.timestep.item() / ctx.scheduler.config.num_train_timesteps)
        self.stored_latents = ctx.latents.clone()
        ctx.latents = ctx.scheduler.add_noise(self.initial_latents.to(ctx.latents.device), self.noise.to(ctx.latents.device), t)
        ctx.latent_model_input = ctx.scheduler.scale_model_input(ctx.latents, ctx.timestep)

        self.stored_conditioning = ctx.inputs.conditioning_data
        ctx.inputs.conditioning_data = self.ref_conditioning

        #set all the processors to store the attention weights
        for attn_processor in self.unet_new_processors:
            attn_processor.store_copy = True

        info(f"timestep_fraction: {timestep_fraction}")

        if self.stop_at >= timestep_fraction:
            #call the unet step to get the attention weights
            info("Running unet to get attention weights")
            ctx.sd_backend.run_unet(ctx, self.dummy_manager, ConditioningMode.Both)
        else:
            info("Skipping unet step to get attention weights")

        #Change back to false, attentions will use the stored maps in the real unet pass
        for attn_processor in self.unet_new_processors:
            attn_processor.store_copy = False
            if self.skip_up_block_1 and 'up_blocks.0.attentions.0' in attn_processor.attn_name and self.skip_until > timestep_fraction:
                attn_processor.store_copy = True
            
        ctx.latents = self.stored_latents
        ctx.timestep = t_orig
        ctx.inputs.conditioning_data = self.stored_conditioning

        if self.once_and_only_once:
            self.and_never_again = True
    
    @callback(ExtensionCallbackType.POST_DENOISE_LOOP)
    def post_denoise_loop(self, ctx: DenoiseContext):
        for attn_processor in self.unet_new_processors:
            attn_processor.saved_query = None
            attn_processor.saved_key = None
            attn_processor.saved_value = None
        torch.cuda.empty_cache()
        

@invocation(
    "RefDrop_extInvocation",
    title="RefDrop Image Reference [Extension]",
    tags=["RefDrop", "reference", "extension"],
    category="latents",
    version="1.0.5",
)
class RefDrop_ExtensionInvocation(BaseInvocation):
    """Incorporates features from the reference image in the output."""
    C: float = InputField(
        title="C",
        description="guidance strength",
        default=0.5,
        ge=-1,
        le=1.0,
    )
    latent_image: LatentsField = InputField(
        title="Latent Image",
        description="Latent image to be targeted.",
    )
    skip_up_block_1: bool = InputField(
        title="Skip Up Block 1",
        description="Skip the first up block. Should help prevent layout bleed",
        default=True
    )
    skip_until: float = InputField(
        title="Skip Until",
        description="Skip the first up block until this timestep",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    positive_conditioning: Optional[Union[ConditioningField, list[ConditioningField]] | None] = InputField(
        title="Positive Conditioning (optional)",
        description="positive condition to pull from reference", ui_order=0,
        default=None,
    )
    negative_conditioning: Optional[Union[ConditioningField, list[ConditioningField]] | None] = InputField(
        title="Negative Conditioning (optional)",
        description="negative condition to avoid from reference", ui_order=1,
        default=None,
    )
    stop_at: float = InputField(
        title="Stop At",
        description="Stop after this timestep",
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    once_and_only_once: bool = InputField(
        title="Once and Only Once",
        description="Compute ONLY for the final step (as determined by Stop At)",
        default=False
    )
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {
            "C": self.C,
            "latent_image_name": self.latent_image.latents_name,
            "skip_up_block_1": self.skip_up_block_1,
            "skip_until": self.skip_until,
            "positive_conditioning": self.positive_conditioning,
            "negative_conditioning": self.negative_conditioning,
            "stop_at": self.stop_at,
            "once_and_only_once": self.once_and_only_once
        }
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="RefDrop",
                extension_kwargs=kwargs
            )
        )