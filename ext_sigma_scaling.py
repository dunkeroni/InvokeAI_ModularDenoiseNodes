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

@guidance_extension_12X("sigma_scaling")
class SigmaScalingGuidance(DenoiseExtensionSD12X):
    """
    Scales sigma based on a piecewise linear function with multiple segments.
    """
    def list_modifies(self) -> dict[str, Callable[..., Any]]:
        return {
            "modify_data_before_denoising": self.modify_data_before_denoising,
        }
    
    def list_swaps(self) -> dict[str, Callable[..., Any]]:
        return super().list_swaps() #REPLACE with {functionname: self.functionname, ...} if you have any swaps
    
    def __post_init__(self, scaling: list[float]):
        self.scaling = scaling
    
    def modify_data_before_denoising(self, data: DenoiseLatentsData) -> DenoiseLatentsData:
        #get sigmas from scheduler
        sigmas = data.scheduler.sigmas
        print(f"sigmas: {sigmas}")
        num_sigmas = len(sigmas)
        #create a piecewise linear function with multiple segments based on self.scaling points and num_sigmas
        scaling = self.scaling
        num_segments = len(scaling) - 1
        segment_size = num_sigmas // num_segments
        for i in range(num_segments):
            start = i * segment_size
            end = (i + 1) * segment_size
            for j in range(start, end):
                sigmas[j] *= scaling[i] + (scaling[i + 1] - scaling[i]) * (j - start) / segment_size
        print(f"modified sigmas: {sigmas}")
        

@invocation(
    "ext_sigma_scaling",
    title="EXT: Sigma Scaling",
    tags=["guidance", "extension", "sigma", "scaling"],
    category="guidance",
    version="1.0.0",
)
class EXT_SigmaGuidanceInvocation(BaseInvocation):
    """
    This is a template for the user-facing node that activates a guidance extension.
    """
    priority: int = InputField(default=500, description="Priority of the guidance module", ui_order=0)
    early_scaling: float = InputField(default=1, ge=0, lt=2, description="Early scaling", ui_order=1)
    mid_scaling: float = InputField(default=1, ge=0, lt=2, description="Mid scaling", ui_order=2)
    late_scaling: float = InputField(default=1, ge=0, lt=2, description="Late scaling", ui_order=3)

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:

        kwargs = dict(
            scaling=[self.early_scaling, self.mid_scaling, self.late_scaling],
        )

        return GuidanceDataOutput(
                guidance_data_output=GuidanceField(
                    guidance_name="sigma_scaling", #matches the decorator name above the guidance class
                    priority=self.priority, #required by all guidance modules
                    extension_kwargs=kwargs, #custom keyword arguments for the guidance module, must be serializeable to JSON
                )
            )