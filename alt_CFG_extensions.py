from typing import Literal, Optional, Union, List

import numpy as np
import torch
from PIL import Image, ImageFilter

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.app.invocations.fields import (
    Input,
    InputField,
    OutputField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext

from .extension_classes import GuidanceField, base_guidance_extension, GuidanceDataOutput

@base_guidance_extension("TCFG")
class TangentialDampingCFG(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
    ):
        super().__init__()
    
    @staticmethod
    def _tangential_cfg(positive_noise_pred: torch.Tensor, negative_noise_pred: torch.Tensor, guidance_scale: float):
        """Implementation of Listing 1 from https://arxiv.org/pdf/2503.18137"""

        all_noise = torch.stack((positive_noise_pred, negative_noise_pred), dim=1).to(dtype=torch.float32)
        all_noise = all_noise.reshape(all_noise.size(0), all_noise.size(1), -1)

        U, S, Vh = torch.linalg.svd(all_noise,full_matrices=False)
        Vh = Vh.to(all_noise.device)
        Vh_modified = Vh.clone().to(all_noise.device)
        Vh_modified[:,1] = 0
        noise_null_flat = negative_noise_pred.reshape(negative_noise_pred.size(0), 1, -1).to(dtype=torch.float32)
        noise_null_flat = noise_null_flat.to(Vh.device)
        x_Vh = torch.matmul(noise_null_flat, Vh.transpose(-2, -1))
        x_Vh_V = torch.matmul(x_Vh, Vh_modified)
        negative_noise_pred = x_Vh_V.reshape(*negative_noise_pred.shape).to(positive_noise_pred.dtype).to(positive_noise_pred.device)
        noise_pred = negative_noise_pred + guidance_scale * (positive_noise_pred - negative_noise_pred)
        return noise_pred

    @callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS)
    def tangential_damped_CFG(self, ctx: DenoiseContext):

        guidance_scale = ctx.inputs.conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[ctx.step_index]

        ctx.noise_pred = self._tangential_cfg(
            ctx.positive_noise_pred,
            ctx.negative_noise_pred,
            guidance_scale,
        )


@invocation(
    "tangential_damping_CFG",
    title="TCFG [Extension]",
    tags=["TFCG", "CFG", "tangential", "extension"],
    category="extension",
    version="1.0.0",
)
class TangentialDampingCFGExtensionInvocation(BaseInvocation):
    """Creates mask for denoising model run."""
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {}
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="TCFG",
                extension_kwargs=kwargs
            )
        )