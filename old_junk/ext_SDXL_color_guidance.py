####################################################################################################
# SDXL Color Guidance
# From: https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
####################################################################################################

from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X
from invokeai.backend.util.logging import info, warning, error
import torch
from typing import Callable, Any, Literal
from invokeai.invocation_api import (
    invocation,
    BaseInvocation,
    InputField,
    InvocationContext,
    Input,
)
from .denoise_latents_extensions import (
    GuidanceField,
    GuidanceDataOutput
)

@guidance_extension_12X("SDXL_color_guidance") #MUST be the same as the guidance_name in the GuidanceField
class SDXLColorGuidance(DenoiseExtensionSD12X):
    """
    Re-normalizes the latents during each step to fit the target VAE space
    """
    def list_modifies(self) -> dict[str, Callable[..., Any]]:
        return {"modify_data_before_scaling": self.modify_data_before_scaling} 
    
    def list_swaps(self) -> dict[str, Callable[..., Any]]:
        return super().list_swaps() #REPLACE with {functionname: self.functionname, ...} if you have any swaps
    
    def __post_init__(self, start_at: float, end_at: float, target_mean: float, channels: list[int]):
        self.start = start_at
        self.end = end_at
        self.target_mean = target_mean
        self.channels = channels

    def modify_data_before_scaling(self, data: DenoiseLatentsData, t: torch.Tensor):
        timevalue = 1 - (t.item() / data.scheduler.config.num_train_timesteps)
        if timevalue >= self.start and timevalue <= self.end:
            channel_shift = 1
            for channel in self.channels:
                data.latents[0, channel] -= (data.latents[0, channel].mean() - self.target_mean) * channel_shift


CHANNEL_SELECTIONS = Literal[
    "All Channels",
    "Colors Only",
    "L0: Brightness",
    "L1: Red->Cyan",
    "L2: Lime->Purple",
    "L3: Structure",
]

CHANNEL_VALUES = {
    "All Channels": [0, 1, 2, 3],
    "Colors Only": [1, 2],
    "L0: Brightness": [0],
    "L1: Red->Cyan": [1],
    "L2: Lime->Purple": [2],
    "L3: Structure": [3],
}

@invocation(
    "ext_SDXL_color_guidance",
    title="EXT: SDXL Color Guidance",
    tags=["guidance", "extension", "Color", "SDXL"],
    category="guidance",
    version="1.0.1",
)
class EXT_SDXLColorGuidanceInvocation(BaseInvocation):
    """
    Fix or apply color drift in SDXL latent space.
    """
    priority: int = InputField(default=500, description="Priority of the guidance module", ui_order=0) #REQUIRED

    start_at: float = InputField(
        title="Start At",
        description="The denoising value at which to start applying color correction. 0 to start at the first step.",
        ge=0,
        lt=1,
        default=0.2,
    )
    end_at: float = InputField(
        title="End At",
        description="The denoising value at which to stop applying color correction. 1 to apply until the last step.",
        gt=0,
        le=1,
        default=1,
    )
    channel_selection: CHANNEL_SELECTIONS = InputField(
        title="Channel Selection",
        description="The channels to affect in the latent correction",
        default="All Channels",
        input=Input.Direct,
    )
    target_mean: float = InputField(
        title="Target Mean",
        description="The target mean to use for the latent correction",
        default=0,
    )

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        channels = CHANNEL_VALUES[self.channel_selection]
        kwargs = {
            "start_at": self.start_at,
            "end_at": self.end_at,
            "target_mean": self.target_mean,
            "channels": channels,
        }

        return GuidanceDataOutput(
                guidance_data_output=GuidanceField(
                    guidance_name="SDXL_color_guidance", #matches the decorator name above the guidance class
                    priority=self.priority, #required by all guidance modules
                    extension_kwargs=kwargs, #custom keyword arguments for the guidance module, must be serializeable to JSON
                )
            )