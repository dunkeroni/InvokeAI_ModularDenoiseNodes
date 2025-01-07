from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X
from invokeai.backend.util.logging import info, warning, error
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
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

@guidance_extension_12X("ddim_eta") #MUST be the same as the guidance_name in the GuidanceField
class DDIMetaGuidance(DenoiseExtensionSD12X):
    """
    Allows manual control over DDIM's "eta" parameter.
    """
    def list_modifies(self) -> dict[str, Callable[..., Any]]:
        return {"modify_data_before_denoising": self.modify_data_before_denoising}
    def list_swaps(self) -> dict[str, Callable[..., Any]]:
        return super().list_swaps() #REPLACE with {functionname: self.functionname, ...} if you have any swaps
    
    def __post_init__(self, eta: float):
        self.eta = eta

    def modify_data_before_denoising(self, data: DenoiseLatentsData):
        if isinstance(data.scheduler, DDIMScheduler):
            data.scheduler_step_kwargs.update({"eta": self.eta})
        else:
            warning("EXT: DDIM eta -> DDIM scheduler not being used, guidance will not be applied.")

@invocation(
    "ext_ddim_eta",
    title="EXT: DDIM eta",
    tags=["guidance", "extension", "DDIM", "eta", "scheduler"],
    category="guidance",
    version="1.0.0",
)
class EXT_DDIMetaGuidanceInvocation(BaseInvocation):
    """
    This is a template for the user-facing node that activates a guidance extension.
    """
    priority: int = InputField(default=500, description="Priority of the guidance module", ui_order=0) #REQUIRED

    eta: float = InputField(default=0, ge=0, le=1, description="DDIM eta value", ui_order=1)

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:

        kwargs = dict(
            eta=self.eta,
        )

        return GuidanceDataOutput(
                guidance_data_output=GuidanceField(
                    guidance_name="ddim_eta", #matches the decorator name above the guidance class
                    priority=self.priority, #required by all guidance modules
                    extension_kwargs=kwargs, #custom keyword arguments for the guidance module, must be serializeable to JSON
                )
            )