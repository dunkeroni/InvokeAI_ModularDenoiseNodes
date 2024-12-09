from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.invocation_api import (
    invocation,
    invocation_output,
    BaseInvocationOutput,
    Input,
    InputField,
    OutputField,
    LatentsOutput,
    InvocationContext,
)
from pydantic import BaseModel
from invokeai.app.invocations.fields import Field
from typing import Type, Any
from invokeai.backend.util.logging import info, warning, error

SD12X_EXTENSIONS = {}

def base_guidance_extension(name: str):
    """Register a guidance extension class object under a string reference"""
    def decorator(cls: Type[ExtensionBase]):
        if name in SD12X_EXTENSIONS:
            raise ValueError(f"Extension {name} already registered")
        info(f"Registered extension {cls.__name__} as {name}")
        SD12X_EXTENSIONS[name] = cls
        return cls
    return decorator

class GuidanceField(BaseModel):
    """Guidance information for extensions in the denoising process."""
    guidance_name: str = Field(description="The name of the guidance extension class")
    #priority: int = Field(default=100, description="Execution order for multiple guidance. Lower numbers go first.")
    extension_kwargs: dict[str, Any] = Field(default={}, description="Keyword arguments for the guidance extension")

@invocation_output("guidance_module_output")
class GuidanceDataOutput(BaseInvocationOutput):
    guidance_data_output: GuidanceField | None = OutputField(
        title="Guidance Module",
        description="Information to alter the denoising process"
    )