##########################################################################################################################
# From: https://arxiv.org/abs/2503.07677
# PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity
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
from typing import Type, Any, Dict, Iterator, List, Optional, Tuple, Union, Literal
from .PLADIS_attention import EntmaxAttention
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode


@base_guidance_extension("PLADIS")
class PLADIS_Guidance(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
        alpha: float,
        lambda_: float,
    ):
        self.alpha = alpha
        self.lambda_ = lambda_
        super().__init__()


    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def pre_denoise_loop(self, ctx: DenoiseContext):
        unet_replacement_processors = {}
        self.unet_new_processors = []
        self.original_processors = ctx.unet.attn_processors

        for key in ctx.unet.attn_processors.keys():
            unet_replacement_processors[key] = EntmaxAttention(self.alpha, self.lambda_)

        ctx.unet.set_attn_processor(unet_replacement_processors)
    
    @callback(ExtensionCallbackType.POST_DENOISE_LOOP)
    def post_denoise_loop(self, ctx: DenoiseContext):
        ctx.unet.set_attn_processor(self.original_processors)
        

ALPHA_OPTIONS = Literal[
    "1.5",
    "2.0",
]

ALPHA_OPTION_LABELS = {
    "1.5": "α=1.5 (α-entmax15)",
    "2.0": "α=2.0 (sparsemax)",
}

@invocation(
    "PLADIS_extInvocation",
    title="PLADIS [Extension]",
    tags=["PLADIS", "attention", "extension"],
    category="extension",
    version="1.0.0",
)
class PLADIS_ExtensionInvocation(BaseInvocation):
    """Applies PLADIS entmax attention."""
    alpha: ALPHA_OPTIONS = InputField(
        default="1.5", description="The alpha value to use", ui_choice_labels=ALPHA_OPTION_LABELS
    )
    lambda_: float = InputField(
        title="lambda",
        description="lambda scale value. Values of 1.5 and 2.0 are allegedly optimal.",
        default=2.0,
        ge=0,
    )
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        alphas = {
            "1.5": 1.5,
            "2.0": 2.0,
        }
        alpha = alphas[self.alpha]
        
        kwargs = {
            "alpha": alpha,
            "lambda_": self.lambda_,
        }
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="PLADIS",
                extension_kwargs=kwargs
            )
        )