from .modular_decorators import module_post_noise_guidance, get_post_noise_guidance_module
from .modular_denoise_latents import Modular_StableDiffusionGeneratorPipeline, ModuleData, PoG_ModuleDataOutput, PoG_ModuleData

import torch
import numpy as np
from typing import Literal, Optional, Callable

from invokeai.app.invocations.baseinvocation import (
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


####################################################################################################
# Gradient Mask
# Adjust mask area based on value of input image
####################################################################################################
@module_post_noise_guidance("gradient_mask")
def apply_gradient_mask(
    self: Modular_StableDiffusionGeneratorPipeline,
    step_output: torch.Tensor,
    step_index: int,
    total_step_count: int,
    t: torch.Tensor,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    """Apply gradient mask to latents."""
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])
    module_id = module_kwargs["module_id"]
    scaling = module_kwargs["scaling"]

    sub_latents = sub_module(
        self,
        step_output,
        step_index,
        total_step_count,
        t,
        sub_module_kwargs,
        **kwargs
    )

    initialized = self.check_persistent_data(module_id, "initialized")
    if initialized is None: #first run
        input_noise = self.context.services.latents.get(self.denoise_node.noise.latents_name).to(step_output.device)
        self.set_persistent_data(module_id, "input_noise", input_noise)
        original_latents_field: LatentsField = self.denoise_node.latents
        original_latents_data = self.context.services.latents.get(original_latents_field.latents_name).to(step_output.device)
        self.set_persistent_data(module_id, "original_latents", original_latents_data) #has shape (1, 4, Y, X)

        mask_image = self.context.services.images.get_pil_image(module_kwargs["mask_image_name"])
        greyscale_mask_latent = torch.from_numpy(np.asarray(mask_image.convert("L"))).to(step_output.device)

        #scale values to be between 0 and 1
        greyscale_mask_latent = greyscale_mask_latent / 255.0
        
        #repeat the mask so it has the same shape as the latents
        greyscale_mask_latent = greyscale_mask_latent.repeat(original_latents_data.shape[0], 1, 1, 1)

        #scale the mask to have the same X and Y as original latents
        greyscale_mask_latent = torch.nn.functional.interpolate(greyscale_mask_latent, size=(original_latents_data.shape[2], original_latents_data.shape[3]), mode="nearest")

        self.set_persistent_data(module_id, "mask_latent", greyscale_mask_latent)

        self.set_persistent_data(module_id, "initialized", True)
    
    original_latents = self.check_persistent_data(module_id, "original_latents")
    mask_latent = self.check_persistent_data(module_id, "mask_latent")
    input_noise = self.check_persistent_data(module_id, "input_noise")

    #1. Create a noised input latent based on the current timestep (skip residual)
    noised_original_latents = self.scheduler.add_noise(original_latents, input_noise, t)

    #2. Create a threshhold mask based on the mask latent and the current timestep
    if scaling == "step":
        threshhold = step_index / total_step_count
    elif scaling == "denoise":
        threshhold = 1 - (t.item() / self.scheduler.config.num_train_timesteps)

    #build mask with threshhold
    mask_bool = (mask_latent < threshhold)

    #3. For each point, if the mask is above the threshhold, use the input latent. Otherwise, use the noise original latent.
    result_latents = torch.where(mask_bool, sub_latents, noised_original_latents)

    return result_latents


@invocation(
    "gradient_mask_module",
    title="Gradient Mask Module",
    tags=["mask", "denoise"],
    category="latents",
    version="1.0.0",
)
class GradientMaskModuleInvocation(BaseInvocation):
    """Apply gradient mask to latents."""

    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="[PoG] SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )

    mask_image: ImageField = InputField(
        default=None,
        title="Mask Image",
        description="The grayscale mask image that is used to create the gradient mask."
    )

    scaling: Literal["step", "denoise"] = InputField(
        default="step",
        title="Scaling",
        description="Scaling of mask. Step scaling scales mask based on step index. Denoise scaling scales mask based on remaining denoise.",
        input=Input.Direct
    )
    
    def invoke(self, context: InvocationContext) -> PoG_ModuleDataOutput:
        return PoG_ModuleDataOutput(
            module_data_output=PoG_ModuleData(
                name="Gradient Mask Module",
                module="gradient_mask",
                module_kwargs={
                    "sub_module": self.sub_module,
                    "module_id": self.id,
                    "mask_image_name": self.mask_image.image_name,
                    "scaling": self.scaling,
                }
            )
        )
