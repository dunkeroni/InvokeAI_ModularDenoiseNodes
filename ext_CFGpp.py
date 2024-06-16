####################################################################################################
# CFG++
# From: https://arxiv.org/pdf/2406.08070 https://github.com/CFGpp-diffusion/CFGpp
####################################################################################################
from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X
from invokeai.backend.util.logging import info, warning, error
import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput
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

@guidance_extension_12X("CFG++") #MUST be the same as the guidance_name in the GuidanceField
class CFGppGuidance(DenoiseExtensionSD12X):
    """
    This is a template for creating a new guidance extension.
    """
    def list_modifies(self) -> dict[str, Callable[..., Any]]:
        return {"modify_data_before_denoising": self.modify_data_before_denoising}
    
    def list_swaps(self) -> dict[str, Callable[..., Any]]:
        return {
            "swap_scheduler_step": self.swap_scheduler_step,
            "swap_combine_noise": self.swap_combine_noise,
            }
    
    def __post_init__(self, cfg_guidance: float, skip_final_step: bool):
        self.cfg_guidance = cfg_guidance
        self.skip_final_step = skip_final_step

    def modify_data_before_denoising(self, data: DenoiseLatentsData):
        self.dataclone = data.copy() # save a copy of the data for later use
        self.scheduler = data.scheduler
        pass
    
    def alpha(self, t):
        t = int(t)
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    def swap_combine_noise(
            self,
            default: Callable,
            unconditioned_next_x: torch.Tensor,
            conditioned_next_x: torch.Tensor,
            guidance_scale: float,
        ) -> torch.Tensor:
        """I'm sad that I have to do this, but I need to save the unconditional noise for the next swap."""
        self.noise_uc = unconditioned_next_x
        return default(unconditioned_next_x, conditioned_next_x, self.cfg_guidance)


    def swap_scheduler_step(
        self,
        default: Callable,
        noise_pred: torch.Tensor,
        timestep: int,
        latents: torch.Tensor,
        **scheduler_step_kwargs,
        ) -> torch.Tensor:
        """Step the scheduler, return the step_output"""
        info(f"CFG++: timestep={timestep}")
        zt = latents
        at = self.alpha(timestep)
        timestep_index = self.scheduler.timesteps.eq(timestep).nonzero().item()
        if timestep > self.scheduler.timesteps[-1]:
            timestep_next = self.scheduler.timesteps[timestep_index + 1]
        else:
            timestep_next = self.scheduler.timesteps[-1]
            if self.skip_final_step:
                return default(noise_pred, timestep, latents, **scheduler_step_kwargs)
        at_next = self.alpha(timestep_next) 

        # tweedie
        z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

        # add noise
        zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * self.noise_uc

        return SchedulerOutput(prev_sample=zt)


@invocation(
    "ext_CFG++",
    title="EXT: CFG++",
    tags=["guidance", "extension", "CFG++"],
    category="guidance",
    version="1.1.0",
)
class EXT_CFGppGuidanceInvocation(BaseInvocation):
    """
    Replaces the default CFG guidance with CFG++
    """
    priority: int = InputField(default=500, description="Priority of the guidance module", ui_order=0) #REQUIRED

    cfg_guidance: float = InputField(default=0.8, description="CFG++ guidance value", title="CFG++", ge=0, ui_order=1)
    skip_final_step: bool = InputField(default=False, description="Skip the final step", title="Skip final step", ui_order=2)

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:

        kwargs = dict(
            cfg_guidance=self.cfg_guidance,
            skip_final_step=self.skip_final_step,
        )

        return GuidanceDataOutput(
                guidance_data_output=GuidanceField(
                    guidance_name="CFG++", #matches the decorator name above the guidance class
                    priority=self.priority, #required by all guidance modules
                    extension_kwargs=kwargs, #custom keyword arguments for the guidance module, must be serializeable to JSON
                )
            )