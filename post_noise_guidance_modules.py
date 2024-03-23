from .modular_decorators import module_post_noise_guidance, get_post_noise_guidance_module
from .modular_denoise_latents import Modular_StableDiffusionGeneratorPipeline, ModuleData, PoG_ModuleDataOutput, PoG_ModuleData

import torch
import numpy as np
from typing import Literal, Optional, Callable

from invokeai.invocation_api import (
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    UIType,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField, LatentsField
from PIL import Image

def resolve_module(module_dict: dict | None) -> tuple[Callable, dict]:
    """Resolve a module from a module dict. Handles None case automatically. """
    if module_dict is None:
        return get_post_noise_guidance_module(None), {}
    else:
        return get_post_noise_guidance_module(module_dict["module"]), module_dict["module_kwargs"]

@module_post_noise_guidance("default_case")
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

