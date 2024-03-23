from .modular_decorators import module_pre_noise_guidance, get_pre_noise_guidance_module
from .modular_denoise_latents import Modular_StableDiffusionGeneratorPipeline, ModuleData, PreG_ModuleDataOutput, PreG_ModuleData

import torch
from typing import Literal, Optional, Callable, Union, List

from invokeai.invocation_api import (
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    UIType,
    invocation,
    ColorField,
)

def resolve_module(module_dict: dict | None) -> tuple[Callable, dict]:
    """Resolve a module from a module dict. Handles None case automatically. """
    if module_dict is None:
        return get_pre_noise_guidance_module(None), {}
    else:
        return get_pre_noise_guidance_module(module_dict["module"]), module_dict["module_kwargs"]


@module_pre_noise_guidance("default_case")
def default_case(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor,
    step_index: int,
    total_step_count: int,
    t: torch.Tensor,
    module_kwargs: dict | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return latents


####################################################################################################
# SD1 Color Guidance
# From: https://github.com/Haoming02/sd-webui-vectorscope-cc
####################################################################################################
@module_pre_noise_guidance("color_offset")
def color_offset(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor,
    step_index: int,
    total_step_count: int,
    t: torch.Tensor,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])
    red = module_kwargs["red"] / total_step_count
    green = module_kwargs["green"] / total_step_count
    blue = module_kwargs["blue"] / total_step_count
    brightness = module_kwargs["brightness"] / total_step_count
    contrast = module_kwargs["contrast"] / total_step_count
    saturation = pow(module_kwargs["saturation"], 1 / total_step_count)
    scaling = module_kwargs["scaling"]

    sub_latents = sub_module(
        self=self,
        latents=latents,
        t=t,
        step_index=step_index,
        total_step_count=total_step_count,
        module_kwargs=sub_module_kwargs,
        **kwargs,
    )

    if scaling == "Linear":
        scale = 1 - (step_index / total_step_count)
    elif scaling == "Denoise":
        scale = (t / self.scheduler.config.num_train_timesteps)
    elif scaling == "None":
        scale = 1
    
    #This could be generated with other methods (randn, etc.), but there doesn't seem to be value in providing those to users
    adjuster = torch.ones_like(sub_latents[:, 0]) * scale
    scaled_saturation = (saturation - 1) * scale + 1

    L = sub_latents[:, 0]
    R = sub_latents[:, 2]
    G = sub_latents[:, 1]
    B = sub_latents[:, 3]

    L += brightness * adjuster
    L += torch.sub(L, torch.mean(L)) * contrast * scale
    R -= red * adjuster
    G += green * adjuster
    B -= blue * adjuster

    R *= scaled_saturation
    G *= scaled_saturation
    B *= scaled_saturation

    return sub_latents

@invocation("color_offset_module_SD1",
    title="Color Offset SD1",
    tags=["modifier", "module", "modular"],
    category="modular",
    version="1.1.2",
)
class ColorOffsetModuleInvocation(BaseInvocation):
    """PreG_MOD: SD1 Color Offset"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="[PreG] SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
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
        default="None",
        input=Input.Direct,
    )

    def invoke(self, context: InvocationContext) -> PreG_ModuleDataOutput:
        #normalize colors from [0, 255] to [-4, 4], then scale by alpha
        red = (self.color.r / 32 - 4) * (self.color.a / 255)
        green = (self.color.g / 32 - 4) * (self.color.a / 255)
        blue = (self.color.b / 32 - 4) * (self.color.a / 255)

        module = PreG_ModuleData(
            name="Color Offset module",
            module="color_offset",
            module_kwargs={
                "sub_module": self.sub_module,
                "red": red,
                "green": green,
                "blue": blue,
                "brightness": self.brightness,
                "contrast": self.contrast,
                "saturation": self.saturation,
                "scaling": self.scaling,
            },
        )

        return PreG_ModuleDataOutput(
            module_data_output=module,
        )



####################################################################################################
# SDXL Color Guidance
# From: https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
####################################################################################################

# Shrinking towards the mean (will also remove outliers)
def soft_clamp_tensor(input_tensor: torch.Tensor, threshold=0.9, boundary=4, channels=[0, 1, 2]):
    for channel in channels:
        channel_tensor = input_tensor[:, channel]
        if not max(abs(channel_tensor.max()), abs(channel_tensor.min())) < 4:
            max_val = channel_tensor.max()
            max_replace = ((channel_tensor - threshold) / (max_val - threshold)) * (boundary - threshold) + threshold
            over_mask = (channel_tensor > threshold)

            min_val = channel_tensor.min()
            min_replace = ((channel_tensor + threshold) / (min_val + threshold)) * (-boundary + threshold) - threshold
            under_mask = (channel_tensor < -threshold)

            input_tensor[:, channel] = torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, channel_tensor))

    return input_tensor

# Center tensor (balance colors)
def shift_tensor(input_tensor, channel_shift=1, channels=[0, 1, 2, 3], target = 0):
    for channel in channels:
        input_tensor[0, channel] -= (input_tensor[0, channel].mean() - target) * channel_shift
    return input_tensor# - input_tensor.mean() * full_shift

# Maximize/normalize tensor
def expand_tensor(input_tensor, boundary=4, channels=[0, 1, 2]):
    for channel in channels:
        input_tensor[0, channel] *= (boundary/2) / input_tensor[0, channel].max()
        #min_val = input_tensor[0, channel].min()
        #max_val = input_tensor[0, channel].max()

        #get min max from 3 standard deviations from mean instead
        mean = input_tensor[0, channel].mean()
        std = input_tensor[0, channel].std()
        min_val = mean - std * 2
        max_val = mean + std * 2

        #colors will always center around 0 for SDXL latents, but brightness/structure will not. Need to adjust this.
        normalization_factor = boundary / max(abs(min_val), abs(max_val))
        input_tensor[0, channel] *= normalization_factor

    return input_tensor

@module_pre_noise_guidance("color_guidance")
def color_guidance(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor,
    step_index: int,
    total_step_count: int,
    t: torch.Tensor,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])
    expand_dynamic_range = False #module_kwargs["expand_dynamic_range"]
    #dynamic_range = module_kwargs["dynamic_range"]
    start_step = module_kwargs["start_step"]
    end_step = module_kwargs["end_step"]
    target_mean = module_kwargs["target_mean"]
    channels = module_kwargs["channels"]
    # expand_dynamic_range: bool = module_kwargs["expand_dynamic_range"]
    timestep: float = t.item()

    sub_latents = sub_module(
        self=self,
        latents=latents,
        t=t,
        step_index=step_index,
        total_step_count=total_step_count,
        module_kwargs=sub_module_kwargs,
        **kwargs,
    )

    if step_index >= start_step and (step_index <= end_step or end_step == -1):
        sub_latents = shift_tensor(sub_latents, 1, channels=channels, target=target_mean)
        # if expand_dynamic_range:
        #     latents = expand_tensor(latents, boundary=dynamic_range, channels=channels)

    return sub_latents

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
    "L1: Cyan->Red": [1],
    "L2: Lime->Purple": [2],
    "L3: Structure": [3],
}

@invocation("color_guidance_module_SDXL",
    title="Color Guidance SDXL",
    tags=["modifier", "module", "modular"],
    category="modular",
    version="1.0.2",
)
class ColorGuidanceModuleInvocation(BaseInvocation):
    """PreG_MOD: Color Guidance (fix SDXL yellow bias)"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="[PreG] SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    start_step: int = InputField(
        title="Start Step",
        description="The step index at which to start applying color correction",
        ge=0,
        default=0,
    )
    end_step: int = InputField(
        title="End Step",
        description="The step index at which to stop applying color correction. -1 to never stop.",
        ge=-1,
        default=-1,
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

    def invoke(self, context: InvocationContext) -> PreG_ModuleDataOutput:

        channels = CHANNEL_VALUES[self.channel_selection]

        module = PreG_ModuleData(
            name="Color Guidance module",
            module="color_guidance",
            module_kwargs={
                "sub_module": self.sub_module,
                "start_step": self.start_step,
                "end_step": self.end_step,
                "target_mean": self.target_mean,
                "channels": channels,
            },
        )

        return PreG_ModuleDataOutput(
            module_data_output=module,
        )


####################################################################################################
# Color Gravitation
# From: strange intuitions
####################################################################################################
"""Takes a list of colors. Before each step, 
    each value in the latent is pulled towards the nearest colors.
    The amount of pull is determined by the gravity and the vector distance to that color."""
@module_pre_noise_guidance("color_gravitation")
@torch.no_grad()
def color_gravitation(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor,
    step_index: int,
    total_step_count: int,
    t: torch.Tensor,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])
    colors: List[ColorField] = module_kwargs["colors"]
    gravity = module_kwargs["gravity"]
    model_type = module_kwargs["model_type"]
    timestep: float = t.item()

    sub_latents = sub_module(
        self=self,
        latents=latents,
        t=t,
        step_index=step_index,
        total_step_count=total_step_count,
        module_kwargs=sub_module_kwargs,
        **kwargs,
    )
    colorvectors: List[torch.Tensor] = []

    for color in colors:
        if model_type == "SD 1.5":
            colortensor = latents.clone() # 1x4x64x64
            colortensor[:, 2, :, :] = -1 * (color.r / 32 - 4) # red
            colortensor[:, 1, :, :] = color.g / 32 - 4 # green
            colortensor[:, 3, :, :] = -1 * (color.b / 32 - 4) # blue

            colorvector = colortensor - latents
            magnitude = torch.norm(colorvector[:, 1:, :, :], dim=1, keepdim=True)
            scaled_colorvector = gravity * (colorvector / (magnitude ** 2)) * color.a / 255
            colorvectors.append(scaled_colorvector)       
        elif model_type == "SDXL":
            raise NotImplementedError("SDXL color gravitation not yet implemented")

    if model_type == "SD 1.5":
        colorvector = torch.sum(torch.stack(colorvectors), dim=0)
        sub_latents += colorvector * (1 - step_index / total_step_count)

    return sub_latents


@invocation("color_gravitation_module",
    title="Color Gravitation",
    tags=["modifier", "module", "modular", "color"],
    category="modular",
    version="1.0.0",
)
class ColorGravitationModuleInvocation(BaseInvocation):
    """PreG_MOD: Color Gravitation"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="[PreG] SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    colors: Union[ColorField, List[ColorField]] = InputField(
        title="Colors",
        description="The colors to use as targets",
        default=[ColorField(r=0, g=0, b=0, a=255)],
    )
    gravity: float = InputField(
        title="Gravity",
        description="The gravity to apply to the chroma",
        default=0.5,
    )
    model_type: Literal["SD 1.5", "SDXL"] = InputField(
        title="Model Type",
        description="The type of model to use",
        default="SD 1.5",
        input=Input.Direct,
    )
    def invoke(self, context: InvocationContext) -> PreG_ModuleDataOutput:
        if isinstance(self.colors, ColorField):
            self.colors = [self.colors]

        module = PreG_ModuleData(
            name="Chroma Gravitation module",
            module="color_gravitation",
            module_kwargs={
                "sub_module": self.sub_module,
                "colors": self.colors,
                "gravity": self.gravity,
                "model_type": self.model_type,
            }
        )

        return PreG_ModuleDataOutput(
            module_data_output=module,
        )