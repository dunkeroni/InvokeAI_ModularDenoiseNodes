from typing import Literal, Optional, Union, List

import numpy as np
import torch
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

from invokeai.backend.util.logging import info, warning, error
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
    """Replaces CFG with TCFG."""
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {}
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="TCFG",
                extension_kwargs=kwargs
            )
        )


@base_guidance_extension("MCG")
class ManualCG(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
        positive_guidance: float,
        negative_guidance: float,
    ):
        super().__init__()
        self.positive_guidance = positive_guidance
        self.negative_guidance = negative_guidance

    @callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS)
    def manual_CG(self, ctx: DenoiseContext):

        guidance_scale = ctx.inputs.conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[ctx.step_index]
        
        ctx.noise_pred = self.positive_guidance * ctx.positive_noise_pred - self.negative_guidance * ctx.negative_noise_pred


@invocation(
    "manual_CG",
    title="MCG [Extension]",
    tags=["MCG", "CFG", "manual", "extension"],
    category="extension",
    version="1.0.0",
)
class ManualCGExtensionInvocation(BaseInvocation):
    """Replaces CFG with MCG."""
    positive_guidance: float = InputField(
        title="Positive Guidance",
        description="Positive guidance value",
        default=7
    )
    negative_guidance: float = InputField(
        title="Negative Guidance",
        description="Negative guidance value",
        default=6
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {
            "positive_guidance": self.positive_guidance,
            "negative_guidance": self.negative_guidance
        }
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="MCG",
                extension_kwargs=kwargs
            )
        )
    

@base_guidance_extension("dCFG")
class DebugCFG(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
    ):
        super().__init__()
        self.alpha_cu= []
        self.alpha_l= []
        self.magc= []
        self.magu= []
        self.magdiff= []
        self.stdc= []
        self.stdu= []

    @callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS)
    def debug_CFG(self, ctx: DenoiseContext):
        conditional = ctx.positive_noise_pred
        unconditional = ctx.negative_noise_pred
        total = ctx.noise_pred
        latents = ctx.latents
        eps = 1e-8

        # Angle between conditional and unconditional guidance
        cosine_similarity = torch.sum(conditional * unconditional, dim=1) / (torch.norm(conditional, dim=1) * torch.norm(unconditional, dim=1) + 1e-8)
        cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
        alphas = torch.acos(cosine_similarity)
        print(alphas.size())
        alpha_cu = torch.mean(torch.abs(alphas))
        info(f"Angle between conditional and unconditional guidance: {alpha_cu}")
        self.alpha_cu.append(alpha_cu.item())

        # Angle between latents and noise prediction
        alpha_l = torch.sum(latents * total, dim=1) / (torch.norm(latents, dim=1) * torch.norm(total, dim=1) + eps)
        alpha_l = torch.clamp(alpha_l, -1, 1)
        alpha_l = torch.acos(alpha_l)
        alpha_l = torch.mean(torch.abs(alpha_l))
        info(f"Angle between latents and noise prediction: {alpha_l}")
        self.alpha_l.append(alpha_l.item())

        # Magnitude of conditional guidance
        magc = torch.norm(conditional)
        info(f"Magnitude of conditional guidance: {magc}")
        self.magc.append(magc.item())

        # Magnitude of difference between conditional and unconditional guidance
        magdiff = torch.norm(conditional - unconditional)
        info(f"Magnitude of difference between conditional and unconditional guidance: {magdiff}")
        self.magdiff.append((magdiff.item()/magc.item())*100)

        # Standard deviation of conditional guidance
        stdc = torch.std(conditional)
        info(f"Standard deviation of conditional guidance: {stdc}")
        self.stdc.append(stdc.item())

    
    @callback(ExtensionCallbackType.POST_DENOISE_LOOP)
    def make_image(self, ctx: DenoiseContext):
        fig, axs = plt.subplots(1, 4, figsize=(30, 8), gridspec_kw={'wspace': 0.2, 'width_ratios': [2, 2, 2, 2]})
        axs[0].plot(self.alpha_cu)
        axs[0].set_title('Angle between conditional and unconditional guidance')
        axs[1].plot(self.alpha_l)
        axs[1].set_title('Angle between latents and noise prediction')
        axs[2].plot(self.magc)
        axs[2].set_title('Magnitude of conditional guidance')
        axs[3].plot(self.magdiff)
        axs[3].set_title("percent difference between cond and unc mags")
        # Save it to /home/dunkeroni/Downloads
        plt.savefig('/home/dunkeroni/Downloads/plot2.png')
        info("Plot saved to /home/dunkeroni/Downloads/plot2.png")


@invocation(
    "debug_CFG",
    title="dCFG [Extension]",
    tags=["dCFG", "CFG", "manual", "extension"],
    category="extension",
    version="1.0.0",
)
class DebugCFGExtensionInvocation(BaseInvocation):
    """Debug CFG Values"""

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {
            #"positive_guidance": self.positive_guidance,
            #"negative_guidance": self.negative_guidance
        }
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="dCFG",
                extension_kwargs=kwargs
            )
        )
    


@base_guidance_extension("RCFG")
class SlerpCFG(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
    ):
        super().__init__()

    @torch.no_grad()
    def rerp(self, t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
        """Rotational linear interpolation between two tensors.
        
        Rotates from v0 towards v1 by t*theta radians, where theta is the angle between them.
        Linearly interpolates the magnitude between ||v0|| and ||v1||.
        
        Args:
            t: Interpolation factor (0 = v0, 1 = v1)
            v0: First tensor (negative/unconditional)
            v1: Second tensor (positive/conditional)
            
        Returns:
            Rotated and magnitude-interpolated tensor
        """
        # Calculate magnitudes of input vectors
        v0_norm = torch.norm(v0, dim=1, keepdim=True)
        v1_norm = torch.norm(v1, dim=1, keepdim=True)
        
        # Normalize vectors
        v0_normalized = v0 / (v0_norm + 1e-6)
        v1_normalized = v1 / (v1_norm + 1e-6)
        
        # Calculate dot product and angle between vectors
        dot_product = torch.sum(v0_normalized * v1_normalized, dim=1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        theta = torch.acos(dot_product)
        
        # Calculate the orthogonal component to v0 in the direction of v1
        # This is similar to Gram-Schmidt orthogonalization
        v1_orthogonal = v1_normalized - dot_product * v0_normalized
        v1_orthogonal_norm = torch.norm(v1_orthogonal, dim=1, keepdim=True)
        v1_orthogonal_normalized = v1_orthogonal / (v1_orthogonal_norm + 1e-6)
        
        # Calculate rotation by t*theta radians
        cos_t_theta = torch.cos(t * theta)
        sin_t_theta = torch.sin(t * theta)
        
        # Rotate v0_normalized towards v1_normalized by t*theta
        rotated_normalized = cos_t_theta * v0_normalized + sin_t_theta * v1_orthogonal_normalized
        
        # Linearly interpolate the magnitude
        target_magnitude = v0_norm + t * (v1_norm - v0_norm)
        
        # Apply the interpolated magnitude to the rotated vector
        result = rotated_normalized * target_magnitude
        
        return result

    @callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS)
    def manual_CG(self, ctx: DenoiseContext):
        guidance_scale = ctx.inputs.conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[ctx.step_index]
        
        ctx.noise_pred = self.rerp(
            guidance_scale,
            ctx.negative_noise_pred,
            ctx.positive_noise_pred,
        )


@invocation(
    "rerp_CFG",
    title="RerpCFG [Extension]",
    tags=["RerpCFG", "CFG", "manual", "extension"],
    category="extension",
    version="1.0.0",
)
class RerpCFGExtensionInvocation(BaseInvocation):
    """Replaces CFG with Rerped CFG."""

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {}
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="RCFG",
                extension_kwargs=kwargs
            )
        )



@base_guidance_extension("ReCFG")
class RemagCFG(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
        remag_scale: float = 4,
        remag_outwards: bool = False,
    ):
        super().__init__()
        self.remag_scale = remag_scale
        self.remag_outwards = remag_outwards

    def remag(self, t: float, x0: torch.Tensor, x1: torch.Tensor, combined: torch.Tensor) -> torch.Tensor:
        x0_norm = torch.norm(x0, dim=1, keepdim=True)
        x1_norm = torch.norm(x1, dim=1, keepdim=True)
        combined_norm = torch.norm(combined, dim=1, keepdim=True)
        
        # Calculate the remag factor
        if not self.remag_outwards:
            remag_factor = x0_norm + t * (x1_norm - x0_norm)
        else:
            minimum_norm = torch.minimum(x0_norm, x1_norm)
            maximum_norm = torch.maximum(x0_norm, x1_norm)
            remag_factor = minimum_norm + t * (maximum_norm - minimum_norm)

        # Normalize the combined tensor
        combined_normalized = combined / (combined_norm + 1e-8)
        combined_normalized = combined_normalized * remag_factor

        # Apply the remag factor to the noise predictions
        return x0 + remag_factor * (x1 - x0)

    @torch.no_grad()
    @callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS)
    def remag_CG(self, ctx: DenoiseContext):
        guidance_scale = ctx.inputs.conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[ctx.step_index]
    
        ctx.noise_pred = self.remag(
            guidance_scale,
            ctx.negative_noise_pred,
            ctx.positive_noise_pred,
            ctx.noise_pred,
        )


@invocation(
    "remag_CFG",
    title="remagCFG [Extension]",
    tags=["remagCFG", "CFG", "manual", "extension"],
    category="extension",
    version="1.1.0",
)
class RemagCFGExtensionInvocation(BaseInvocation):
    """Replaces CFG with reMag CFG."""
    remag_scale: float = InputField(
        title="Remag Scale",
        description="Remag scale value",
        default=4
    )
    outwards: bool = InputField(
        title="Outwards",
        description="Whether to use outwards remag",
        default=False
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {
            "remag_scale": self.remag_scale,
            "remag_outwards": self.outwards
        }
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="ReCFG",
                extension_kwargs=kwargs
            )
        )