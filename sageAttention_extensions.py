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
from .sage_attention import SageAttention
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage
from contextlib import contextmanager

@base_guidance_extension("SageAttention")
class SageAttention_Guidance(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
    ):
        super().__init__()

    @contextmanager
    def patch_unet(unet: UNet2DConditionModel, original_weights: OriginalWeightsStorage):
        """A context manager that patches `unet` with the provided attention processor."""

        unet_orig_processors = unet.attn_processors
        unet_replacement_processors = {}
        for key in unet.attn_processors.keys():
            unet_replacement_processors[key] = SageAttention()

        try:
            unet.set_attn_processor(unet_replacement_processors)
            yield None

        finally:
            unet.set_attn_processor(unet_orig_processors)
        

@invocation(
    "SageAttention_extInvocation",
    title="SageAttention [Extension]",
    tags=["SageAttention", "attention", "gottagofast"],
    category="extensions",
    version="1.0.0",
)
class SageAttention_ExtensionInvocation(BaseInvocation):
    """Incorporates features from the reference image in the output."""
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {}
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="SageAttention",
                extension_kwargs=kwargs
            )
        )