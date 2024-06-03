from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X
import torch
from typing import Callable, Any
from invokeai.invocation_api import (
    invocation,
    BaseInvocation,
    InputField,
    InvocationContext,
)
from .denoise_latents_extensions import (
    GuidanceField,
    GuidanceDataOutput
)

@guidance_extension_12X("cfg_rescale")
class CfgRescaleGuidance(DenoiseExtensionSD12X):

    def list_modifies(self) -> dict[str, Callable[..., Any]]:
        return super().list_modifies()
    
    def list_swaps(self) -> dict[str, Callable[..., Any]]:
        return {
            "swap_combine_noise": self.swap_combine_noise
        }
    
    def __post_init__(self, enabled: bool, rescale_multiplier: float):
        self.enabled = enabled
        self.rescale_multiplier = rescale_multiplier
    
    def _rescale_cfg(self, total_noise_pred: torch.Tensor, pos_noise_pred: torch.Tensor, multiplier: float):
        """Implementation of Algorithm 2 from https://arxiv.org/pdf/2305.08891.pdf."""
        ro_pos = torch.std(pos_noise_pred, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(total_noise_pred, dim=(1, 2, 3), keepdim=True)

        x_rescaled = total_noise_pred * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * total_noise_pred
        return x_final

    def swap_combine_noise(
        self,
        default: Callable,
        unconditioned_next_x: torch.Tensor,
        conditioned_next_x: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Wrap the standard noise prediction to rescale the result based on the conditioned result"""
        noise_pred = default(unconditioned_next_x, conditioned_next_x, guidance_scale)
        if self.enabled:
            noise_pred = self._rescale_cfg(
                total_noise_pred=noise_pred,
                pos_noise_pred=conditioned_next_x,
                multiplier=self.rescale_multiplier)
        return noise_pred

@invocation(
    "ext_cfg_rescale",
    title="EXT: CFG Rescale",
    tags=["guidance", "extension", "CFG", "Rescale"],
    category="guidance",
    version="1.0.0",
)
class EXT_CFGRescaleGuidanceInvocation(BaseInvocation):
    priority: int = InputField(default=500, description="Priority of the guidance module", ui_order=0, ui_hidden=True)
    enabled: bool = InputField(default=True, description="Enable rescale guidance", ui_order=1)
    rescale_multiplier: float = InputField(ge=0, lt=1, default=0.7, description="Rescale multiplier", ui_order=2)

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:

        kwargs = dict(
            enabled=self.enabled,
            rescale_multiplier=self.rescale_multiplier
        )

        return GuidanceDataOutput(
                guidance_data_output=GuidanceField(
                    guidance_name="cfg_rescale",
                    priority=self.priority,
                    extension_kwargs=kwargs,
                )
            )