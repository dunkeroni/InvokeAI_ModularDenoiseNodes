####################################################################################################
# SD1 Color Guidance
# From: https://github.com/Haoming02/sd-webui-vectorscope-cc
####################################################################################################

from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X
import torch
from typing import Callable, Any, Literal
from invokeai.invocation_api import (
    invocation,
    BaseInvocation,
    InputField,
    InvocationContext,
    ColorField,
    Input,
)
from .denoise_latents_extensions import (
    GuidanceField,
    GuidanceDataOutput
)

@guidance_extension_12X("SD12_color_offset")
class SD12ColorOffsetGuidance(DenoiseExtensionSD12X):
    """
    Adjusts the color of the latent by an ammount based on the step index.
    """
    def list_modifies(self) -> dict[str, Callable[..., Any]]:
        return {"modify_data_before_scaling": self.modify_data_before_scaling}
    
    def list_swaps(self) -> dict[str, Callable[..., Any]]:
        return super().list_swaps()
    
    def __post_init__(self, red: float, green: float, blue: float, brightness: float, contrast: float, saturation: float, scaling: str):
        self.red = red / self.input_data.steps
        self.green = green / self.input_data.steps
        self.blue = blue / self.input_data.steps
        self.brightness = brightness / self.input_data.steps
        self.contrast = contrast / self.input_data.steps
        self.saturation = pow(saturation, 1 / self.input_data.steps)
        self.scaling = scaling

    def modify_data_before_scaling(self, data: DenoiseLatentsData, t: torch.Tensor):
        if self.scaling == "Linear":
            scale = 1 - (data.step_index / self.input_data.steps)
        elif self.scaling == "Denoise":
            scale = (t / data.scheduler.config.num_train_timesteps)
        elif self.scaling == "None":
            scale = 1
        
        #This could be generated with other methods (randn, etc.), but there doesn't seem to be value in providing those to users
        adjuster = torch.ones_like(data.latents[:, 0]) * scale
        scaled_saturation = (self.saturation - 1) * scale + 1

        L = data.latents[:, 0] # latents is [1, 4, Y, X], this gets the [1, 0, Y, X] slice
        R = data.latents[:, 2]
        G = data.latents[:, 1]
        B = data.latents[:, 3]

        #Slices are references to sub-tensors, so we can modify them in place
        L += self.brightness * adjuster
        L += torch.sub(L, torch.mean(L)) * self.contrast * scale
        R -= self.red * adjuster
        G += self.green * adjuster
        B -= self.blue * adjuster

        R *= scaled_saturation
        G *= scaled_saturation
        B *= scaled_saturation

@invocation(
    "ext_SD_1_2_ColorOffset",
    title="EXT: SD 1/2 Color Offset",
    tags=["guidance", "extension", "color", "offset"],
    category="guidance",
    version="1.0.0",
)
class EXT_SD12ColorOffsetGuidanceInvocation(BaseInvocation):
    """
    Biases the generated image towards a specific color.
    """
    priority: int = InputField(default=500, description="Priority of the guidance module", ui_order=0, ui_hidden=True)
    color: ColorField = InputField(
        title="Color",
        description="The color to use for the offset",
        default=ColorField(r=1, g=1, b=1, a=255),
    )
    brightness: float = InputField(
        title="Brightness Offset",
        description="[-6 - 6] The amount to shift the brightness",
        ge=-6,
        le=6,
        default=0,
    )
    contrast: float = InputField(
        title="Contrast Offset",
        description="[-5 - 5] The amount to shift the contrast",
        ge=-5,
        le=5,
        default=0,
    )
    saturation: float = InputField(
        title="Saturation Target",
        description="[-0.2 - 1.8] The target for the saturation",
        ge=0.2,
        le=1.8,
        default=1,
    )
    scaling: Literal["Linear", "Denoise", "None"] = InputField(
        title="Scaling",
        description="The scaling method to use",
        default="Denoise",
        input=Input.Direct,
    )

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        #normalize colors from [0, 255] to [-4, 4], then scale by alpha
        red = (self.color.r / 32 - 4) * (self.color.a / 255)
        green = (self.color.g / 32 - 4) * (self.color.a / 255)
        blue = (self.color.b / 32 - 4) * (self.color.a / 255)

        kwargs = {
            "red": red,
            "green": green,
            "blue": blue,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "scaling": self.scaling,
        }

        return GuidanceDataOutput(
                guidance_data_output=GuidanceField(
                    guidance_name="SD12_color_offset", #matches the decorator name above the guidance class
                    priority=self.priority, #required by all guidance modules
                    extension_kwargs=kwargs, #custom keyword arguments for the guidance module, must be serializeable to JSON
                )
            )