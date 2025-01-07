from typing import Literal, Optional, Union, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    OutputField,
)
from invokeai.app.invocations.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.model import UNetField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager import LoadedModel
from invokeai.backend.model_manager.config import MainConfigBase, ModelVariantType
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from torchvision.transforms.functional import resize as tv_resize
from invokeai.backend.util.devices import TorchDevice
from torch import Tensor
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.stable_diffusion.extensions.inpaint import InpaintExt

from .extension_classes import GuidanceField, base_guidance_extension



@invocation_output("gradient_mask_extension_output")
class GradientMaskExtensionOutput(BaseInvocationOutput):
    """Outputs a denoise mask and an image representing the total gradient of the mask."""

    mask_extension: GuidanceField = OutputField(
        description="Guidance Extension for masked denoise",
    )
    expanded_mask_area: ImageField = OutputField(
        description="Image representing the total gradient area of the mask. For paste-back purposes."
    )



@base_guidance_extension("InpaintMaskGuidance")
class InpaintMaskGuidance(InpaintExt):
    def __init__(
        self,
        context: InvocationContext,
        mask_name: str,
        is_gradient_mask: bool,
    ):
        """Initialize InpaintExt.
        This override is purely to adapt the Invoke internal extension to accept the mask_name as a string.
        """
        super(InpaintExt,self).__init__() # skip the super call to the InvokeAI version
        self._mask = context.tensors.load(mask_name)
        self._is_gradient_mask = is_gradient_mask
        self._noise: Optional[torch.Tensor] = None

    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def init_tensors(self, ctx: DenoiseContext):
        self._mask = tv_resize(self._mask, ctx.latents.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
        super().init_tensors(ctx)



@invocation(
    "gradient_mask_extension",
    title="Gradient Mask [Extension]",
    tags=["mask", "denoise", "extension"],
    category="extension",
    version="1.4.0",
)
class GradientMaskExtensionInvocation(BaseInvocation):
    """Creates mask for denoising model run."""

    mask: ImageField = InputField(default=None, description="Image which will be masked", ui_order=1)
    edge_radius: int = InputField(
        default=16, ge=0, description="How far to blur/expand the edges of the mask", ui_order=2
    )
    coherence_mode: Literal["Gaussian Blur", "Box Blur", "Staged"] = InputField(default="Gaussian Blur", ui_order=3)
    minimum_denoise: float = InputField(
        default=0.0, ge=0, le=1, description="Minimum denoise level for the coherence region", ui_order=4
    )
    image: Optional[ImageField] = InputField(
        default=None,
        description="OPTIONAL: Only connect for specialized Inpainting models, masked_latents will be generated from the image with the VAE",
        title="[OPTIONAL] Image",
        ui_order=6,
    )
    unet: Optional[UNetField] = InputField(
        description="OPTIONAL: If the Unet is a specialized Inpainting model, masked_latents will be generated from the image with the VAE",
        default=None,
        input=Input.Connection,
        title="[OPTIONAL] UNet",
        ui_order=5,
    )
    vae: Optional[VAEField] = InputField(
        default=None,
        description="OPTIONAL: Only connect for specialized Inpainting models, masked_latents will be generated from the image with the VAE",
        title="[OPTIONAL] VAE",
        input=Input.Connection,
        ui_order=7,
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled, ui_order=8)
    fp32: bool = InputField(
        default=False,
        description=FieldDescriptions.fp32,
        ui_order=9,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GradientMaskExtensionOutput:
        mask_image = context.images.get_pil(self.mask.image_name, mode="L")
        if self.edge_radius > 0:
            if self.coherence_mode == "Box Blur":
                blur_mask = mask_image.filter(ImageFilter.BoxBlur(self.edge_radius))
            else:  # Gaussian Blur OR Staged
                # Gaussian Blur uses standard deviation. 1/2 radius is a good approximation
                blur_mask = mask_image.filter(ImageFilter.GaussianBlur(self.edge_radius / 2))

            blur_tensor: torch.Tensor = image_resized_to_grid_as_tensor(blur_mask, normalize=False)

            # redistribute blur so that the original edges are 0 and blur outwards to 1
            blur_tensor = (blur_tensor - 0.5) * 2
            blur_tensor[blur_tensor < 0] = 0.0

            threshold = 1 - self.minimum_denoise

            if self.coherence_mode == "Staged":
                # wherever the blur_tensor is less than fully masked, convert it to threshold
                blur_tensor = torch.where((blur_tensor < 1) & (blur_tensor > 0), threshold, blur_tensor)
            else:
                # wherever the blur_tensor is above threshold but less than 1, drop it to threshold
                blur_tensor = torch.where((blur_tensor > threshold) & (blur_tensor < 1), threshold, blur_tensor)

        else:
            blur_tensor: torch.Tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)

        mask_name = context.tensors.save(tensor=blur_tensor.unsqueeze(1))

        # compute a [0, 1] mask from the blur_tensor
        expanded_mask = torch.where((blur_tensor < 1), 0, 1)
        resized_expanded_mask = tv_resize(expanded_mask, (expanded_mask.shape[-2] // 8, expanded_mask.shape[-1] // 8), T.InterpolationMode.BILINEAR, antialias=False)
        expanded_mask = torch.where((resized_expanded_mask < 1), 0, 1)
        upscaled_expanded_mask = tv_resize(expanded_mask, (expanded_mask.shape[-2] * 8, expanded_mask.shape[-1] * 8), T.InterpolationMode.NEAREST, antialias=False)
        expanded_mask_image = Image.fromarray((upscaled_expanded_mask.squeeze(0).numpy() * 255).astype(np.uint8), mode="L")
        expanded_image_dto = context.images.save(expanded_mask_image)

        masked_latents_name = None
        if self.unet is not None and self.vae is not None and self.image is not None:
            # all three fields must be present at the same time
            main_model_config = context.models.get_config(self.unet.unet.key)
            assert isinstance(main_model_config, MainConfigBase)
            if main_model_config.variant is ModelVariantType.Inpaint:
                mask = blur_tensor
                vae_info: LoadedModel = context.models.load(self.vae.vae)
                image = context.images.get_pil(self.image.image_name)
                image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                img_mask = tv_resize(mask, image_tensor.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
                masked_image = image_tensor * torch.where(img_mask < 0.5, 0.0, 1.0)
                context.util.signal_progress("Running VAE encoder")
                masked_latents = ImageToLatentsInvocation.vae_encode(
                    vae_info, self.fp32, self.tiled, masked_image.clone()
                )
                masked_latents_name = context.tensors.save(tensor=masked_latents)

        return GradientMaskExtensionOutput(
            mask_extension=GuidanceField(guidance_name="InpaintMaskGuidance", extension_kwargs={"mask_name": mask_name, "is_gradient_mask": True}),
            expanded_mask_area=ImageField(image_name=expanded_image_dto.image_name),
        )





# @invocation(
#     "create_gradient_mask_v2",
#     title="Gradient Mask V2",
#     tags=["mask", "denoise"],
#     category="latents",
#     version="2.0.0",
# )
# class CreateGradientMaskV2Invocation(BaseInvocation):
#     """Creates mask for denoising model run."""

#     mask: Union[ImageField, List[ImageField]] = InputField(default=None, description="Image which will be masked", ui_order=1)
#     max_mask_expansion: int = InputField(
#         default=24, ge=0, multiple_of=8, description="How far to expand the edges of the mask", ui_order=2
#     )
#     minimum_denoise: float = InputField(
#         default=0.0, ge=0, le=1, description="Minimum denoise level for the coherence region", ui_order=4
#     )
#     latent_scale: bool = InputField(default=True, description="Scale the mask to the latent size before processing", ui_order=5, ui_hidden=True)
#     process_on_device: bool = InputField(default=False, description="Process the mask on the same device as inference (GPU, typically)", ui_order=6)

#     @torch.no_grad()
#     def invoke(self, context: InvocationContext) -> GradientMaskOutput:
#         if not isinstance(self.mask, list):
#             mask = [self.mask]
#         else:
#             mask = self.mask
        
#         mask_images = [context.images.get_pil(m.image_name, mode="L") for m in mask]

#         #convert to tensors and combine, keeping the lowest value of each
#         tensor_images = [image_resized_to_grid_as_tensor(m, normalize=False) for m in mask_images]
#         tensor_images = [tv_resize(m, tensor_images[0].shape[-2:], T.InterpolationMode.BILINEAR, antialias=False) for m in tensor_images]
#         mask_tensor = torch.stack(tensor_images, dim=0).min(dim=0)
#         mask_tensor = mask_tensor.values / 255.0

#         #downscale by a factor fo 8 to match the latent size
#         if self.latent_scale:
#             mask_tensor = tv_resize(mask_tensor.values, [s // LATENT_SCALE_FACTOR for s in mask_tensor.shape[-2:]], T.InterpolationMode.BILINEAR, antialias=False)
#             expansion_count = self.max_mask_expansion // LATENT_SCALE_FACTOR
#         else:
#             expansion_count = self.max_mask_expansion
        
#         #expansion steps are linearly spaced between 0 and 1
#         expansion_steps = torch.linspace(0, 1, expansion_count + 1).flip(0)

#         #investigating for speed improvement
#         if self.process_on_device:
#             mask_tensor = mask_tensor.to(TorchDevice.choose_torch_device())
#             expansion_steps = expansion_steps.to(TorchDevice.choose_torch_device())
#         device = mask_tensor.device
        

#         #expand the mask
#         # We are using a convolution interaction to expand darker areas of the mask to the lighter areas
#         # The input mask(s) may not be binary, and could already be gradients.
#         # We start by inverting the mask so white is full denoise and black is fully preserved.
#         # We split the convolution into multiple steps on binned values so that we can apply a different expansion factor to each step
#         # This allows us to have a more gradual expansion of the mask
#         mask_tensor = 1 - mask_tensor
#         combine_mask_bool_tensor = (mask_tensor >= expansion_steps[0]) # catches 100% regions
        
#         for i in range(expansion_count):
#             #create a boolean mask for the current expansion_step values bin
#             mask_bin_bool_tensor = (mask_tensor >= expansion_steps[i+1]) & (mask_tensor < expansion_steps[i])
#             mask_bin_tensor = torch.where(mask_bin_bool_tensor, mask_tensor, torch.zeros_like(mask_tensor))
#             mask_bin_tensor = mask_bin_tensor.unsqueeze(0).unsqueeze(0)
#             #apply the convolution, dilate by 1
#             expanded_mask_bin_tensor = torch.nn.functional.conv2d(mask_bin_tensor, torch.ones(1, 1, 3, 3).to(device), padding=1)
#             #set newly expanded regions to be the next expansion step
#             mask_tensor = torch.where(mask_bin_bool_tensor, expanded_mask_bin_tensor, mask_tensor)



        


#         masked_latents_name = context.tensors.save(tensor=masked_latents)

#         return GradientMaskOutput(
#             denoise_mask=DenoiseMaskField(mask_name=mask_name, masked_latents_name=masked_latents_name, gradient=True),
#             expanded_mask_area=ImageField(image_name=expanded_image_dto.image_name),
#         )
