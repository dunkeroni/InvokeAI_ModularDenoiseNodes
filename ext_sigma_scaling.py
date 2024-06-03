from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X
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
    
    def modify_data_before_denoising(self, data: DenoiseLatentsData):
        #get sigmas from scheduler
        sigmas = data.scheduler.sigmas
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
        

@invocation(
    "ext_sigma_scaling",
    title="EXT: Sigma Scaling",
    tags=["guidance", "extension", "sigma", "scaling"],
    category="guidance",
    version="1.1.0",
)
class EXT_SigmaGuidanceInvocation(BaseInvocation):
    """
    This is a template for the user-facing node that activates a guidance extension.
    """
    priority: int = InputField(default=500, description="Priority of the guidance module", ui_order=0)
    scaling_point_1: float = InputField(default=1, ge=0, lt=2, description="Scaling at the start of the process", ui_order=1)
    scaling_point_2: float = InputField(default=1, ge=0, lt=2, description="At 25%", ui_order=2)
    scaling_point_3: float = InputField(default=1, ge=0, lt=2, description="At 50%", ui_order=3)
    scaling_point_4: float = InputField(default=1, ge=0, lt=2, description="At 75%", ui_order=4)
    scaling_point_5: float = InputField(default=1, ge=0, lt=2, description="Scaling at the final steps", ui_order=5)

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:

        kwargs = dict(
            scaling=[self.scaling_point_1, self.scaling_point_2, self.scaling_point_3, self.scaling_point_4, self.scaling_point_5]
        )

        return GuidanceDataOutput(
                guidance_data_output=GuidanceField(
                    guidance_name="sigma_scaling", #matches the decorator name above the guidance class
                    priority=self.priority, #required by all guidance modules
                    extension_kwargs=kwargs, #custom keyword arguments for the guidance module, must be serializeable to JSON
                )
            )