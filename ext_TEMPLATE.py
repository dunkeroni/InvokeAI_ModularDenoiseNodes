from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X
from invokeai.backend.util.logging import info, warning, error
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

@guidance_extension_12X("template_unique_name") #MUST be the same as the guidance_name in the GuidanceField
class TemplateGuidance(DenoiseExtensionSD12X):
    """
    This is a template for creating a new guidance extension.
    """
    def list_modifies(self) -> dict[str, Callable[..., Any]]:
        return super().list_modifies() #REPLACE with {functionname: self.functionname, ...} if you have any modifies
    
    def list_swaps(self) -> dict[str, Callable[..., Any]]:
        return super().list_swaps() #REPLACE with {functionname: self.functionname, ...} if you have any swaps
    
    def __post_init__(self, enabled: bool):
        self.enabled = enabled

@invocation(
    "ext_template_rename_me",
    title="EXT: TEMPLATE",
    tags=["guidance", "extension", "template", "rename me"],
    category="guidance",
    version="1.0.0",
)
class EXT_TemplateGuidanceInvocation(BaseInvocation):
    """
    This is a template for the user-facing node that activates a guidance extension.
    """
    priority: int = InputField(default=500, description="Priority of the guidance module", ui_order=0) #REQUIRED

    enabled: bool = InputField(default=True, description="Enable the guidance module", ui_order=1) #EXAMPLE

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:

        kwargs = dict(
            enabled=self.enabled, #EXAMPLE PARAMETER
        )

        return GuidanceDataOutput(
                guidance_data_output=GuidanceField(
                    guidance_name="template_unique_name", #matches the decorator name above the guidance class
                    priority=self.priority, #required by all guidance modules
                    extension_kwargs=kwargs, #custom keyword arguments for the guidance module, must be serializeable to JSON
                )
            )