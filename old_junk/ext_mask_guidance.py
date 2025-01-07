from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X

import einops
import torch
from typing import Callable
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torchvision.transforms.functional import resize as tv_resize

from typing import Literal, Optional
from invokeai.invocation_api import (
    invocation,
    BaseInvocation,
    InputField,
    Input,
    ImageField,
    VAEField,
    UNetField,
    FieldDescriptions,
    InvocationContext,
    invocation_output,
    OutputField,
    BaseInvocationOutput
)
from .denoise_latents_extensions import (
    GuidanceField,
)
import torchvision.transforms as T
import numpy as np
from invokeai.backend.model_manager import LoadedModel
from invokeai.backend.model_manager.config import MainConfigBase, ModelVariantType
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.app.invocations.image_to_latents import ImageToLatentsInvocation

from PIL import Image, ImageFilter

from invokeai.backend.util.devices import TorchDevice
DEFAULT_PRECISION = TorchDevice.choose_torch_dtype()

@guidance_extension_12X("mask_guidance")
class MaskGuidance(DenoiseExtensionSD12X):

    def list_modifies(self) -> dict[str, Callable]:
        return {
            "modify_data_before_denoising": self.modify_data_before_denoising,
            "modify_data_before_scaling": self.modify_data_before_scaling,
            "modify_data_before_noise_prediction": self.modify_data_before_noise_prediction,
            "modify_result_before_callback": self.modify_result_before_callback,
            "modify_data_after_denoising": self.modify_data_after_denoising,
            }
    
    def list_swaps(self) -> dict[str, Callable]:
        return super().list_swaps()

    def __post_init__(self, mask_name: str, masked_latents_name: str | None, gradient_mask: bool):
        """Load inputs"""
        self.mask: torch.Tensor = self.context.tensors.load(mask_name)
        self.masked_latents = None if masked_latents_name is None else self.context.tensors.load(masked_latents_name)
        self.gradient_mask: bool = gradient_mask
        self.unet_type: str = self.input_data.unet.unet.base
    
    def modify_data_before_denoising(self, data: DenoiseLatentsData):
        """Store relevant data for later use, create noise if necessary"""
        self.scheduler: SchedulerMixin = data.scheduler
        self.inpaint_model: bool = data.unet.conv_in.in_channels == 9
        self.seed: int = data.seed
        self.orig_latents: torch.Tensor = data.latents.clone()

        if data.noise is not None:
            self.noise = data.noise.clone()
        else:
            self.noise = torch.randn(
                self.orig_latents.shape,
                dtype=torch.float32,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(data.seed),
            ).to(device=self.orig_latents.device, dtype=self.orig_latents.dtype)
        
        self.mask = tv_resize(self.mask, list(self.orig_latents.shape[-2:]))
        self.mask = self.mask.to(device=self.orig_latents.device, dtype=self.orig_latents.dtype)

    def mask_from_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Create a mask based on the current timestep"""
        if self.inpaint_model:
            mask_bool = self.mask < 1
            floored_mask = torch.where(mask_bool, 0, 1)
            return floored_mask
        elif self.gradient_mask:
            threshhold = (t.item()) / self.scheduler.config.num_train_timesteps
            mask_bool = self.mask < 1 - threshhold
            timestep_mask = torch.where(mask_bool, 0, 1)
            return timestep_mask.to(device=self.mask.device)
        else:
            return self.mask.clone()

    def modify_data_before_scaling(self, data: DenoiseLatentsData, t: torch.Tensor):
        """Replace unmasked region with original latents. Called before the scheduler scales the latent values."""
        if self.inpaint_model:
            return # skip this stage

        latents = data.latents
        #expand to match batch size if necessary
        batch_size = latents.size(0)
        mask = self.mask_from_timestep(t).to(device=latents.device, dtype=latents.dtype)
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if t.dim() == 0:
            t = einops.repeat(t, "-> batch", batch=batch_size)

        # create noised version of the original latents
        noised_latents = self.scheduler.add_noise(self.orig_latents, self.noise, t)
        noised_latents = einops.repeat(noised_latents, "b c h w -> (repeat b) c h w", repeat=batch_size).to(device=latents.device, dtype=latents.dtype)
        mask = self.mask_from_timestep(t).to(device=latents.device, dtype=latents.dtype)
        self.last_mask = mask #store for callback visual
        masked_input = torch.lerp(latents, noised_latents, mask)

        data.latents = masked_input

    def shrink_mask(self, mask: torch.Tensor, n_operations: int) -> torch.Tensor:
        kernel = torch.ones(1, 1, 3, 3).to(device=mask.device, dtype=mask.dtype)
        for _ in range(n_operations):
            mask = torch.nn.functional.conv2d(mask, kernel, padding=1).clamp(0, 1)
        return mask

    def modify_data_before_noise_prediction(self, data: DenoiseLatentsData, t: torch.Tensor):
        """Expand latents with information needed by inpaint model"""
        if not self.inpaint_model:
            return # skip this stage

        latents = data.latents
        mask = self.mask_from_timestep(t).to(device=latents.device, dtype=latents.dtype)
        if self.masked_latents is None:
            #latent values for a black region after VAE encode
            if self.unet_type == "sd-1":
                latent_zeros = [0.78857421875, -0.638671875, 0.576171875, 0.12213134765625]
            elif self.unet_type == "sd-2":
                latent_zeros = [0.7890625, -0.638671875, 0.576171875, 0.12213134765625]
                print("WARNING: SD-2 Inpaint Models are not yet supported")
            elif self.unet_type == "sdxl":
                latent_zeros = [-0.578125, 0.501953125, 0.59326171875, -0.393798828125]
            else:
                raise ValueError(f"Unet type {self.unet_type} not supported as an inpaint model. Where did you get this?")

            # replace masked region with specified values
            mask_values = torch.tensor(latent_zeros).view(1, 4, 1, 1).expand_as(latents).to(device=latents.device, dtype=latents.dtype)
            small_mask = self.shrink_mask(mask, 1) #make the synthetic mask fill in the masked_latents smaller than the mask channel
            self.masked_latents = torch.where(small_mask == 0, mask_values, self.orig_latents)

        masked_latents = self.scheduler.scale_model_input(self.masked_latents,t)
        masked_latents = einops.repeat(masked_latents, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        model_input = torch.cat([latents, 1 - mask, masked_latents], dim=1).to(dtype=latents.dtype, device=latents.device)

        data.latents = model_input

    def modify_result_before_callback(self, step_output, data, t):
        """Fix preview images to show the original image in the unmasked region"""
        if hasattr(step_output, "denoised"): #LCM Sampler
            prediction = step_output.denoised
        elif hasattr(step_output, "pred_original_sample"): #Samplers with final predictions
            prediction = step_output.pred_original_sample
        else: #all other samplers (no prediction available)
            prediction = step_output.prev_sample

        mask = self.last_mask
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=prediction.size(0))
        step_output.pred_original_sample = torch.lerp(prediction, self.orig_latents.to(dtype=prediction.dtype), mask.to(dtype=prediction.dtype))


    def modify_data_after_denoising(self, data: DenoiseLatentsData):
        """Apply original unmasked to denoised latents"""
        if self.inpaint_model:
            if self.masked_latents is None:
                mask = self.shrink_mask(self.mask, 1)
            else:
                return 
        else:
            mask = self.mask_from_timestep(torch.Tensor([0]))
        latents = data.latents
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        latents = torch.lerp(latents, self.orig_latents.to(dtype=latents.dtype), mask.to(dtype=latents.dtype)).to(device=latents.device)

        data.latents = latents


@invocation_output("ext_gradient_mask_output")
class GradientMaskExtensionOutput(BaseInvocationOutput):
    """Outputs a denoise mask and an image representing the total gradient of the mask."""

    guidance_module: GuidanceField = OutputField(
        title="Guidance Module",
        description="Information to alter the denoising process"
    )
    expanded_mask_area: ImageField = OutputField(
        description="Image representing the total gradient area of the mask. For paste-back purposes."
    )

@invocation(
    "ext_create_gradient_mask",
    title="EXT: Gradient Mask",
    tags=["guidance", "extension", "mask", "denoise"],
    category="guidance",
    version="1.1.0",
)
class EXT_GradientMaskInvocation(BaseInvocation):
    """Creates mask for denoising model run."""
    priority: int = InputField(default=100, ge=0, description="Priority of the guidance module", ui_order=0)
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
        default=DEFAULT_PRECISION == "float32",
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
        expanded_mask_image = Image.fromarray((expanded_mask.squeeze(0).numpy() * 255).astype(np.uint8), mode="L")
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
                masked_latents = ImageToLatentsInvocation.vae_encode(
                    vae_info, self.fp32, self.tiled, masked_image.clone()
                )
                masked_latents_name = context.tensors.save(tensor=masked_latents)

        kwargs = dict(
            mask_name=mask_name,
            masked_latents_name=masked_latents_name,
            gradient_mask=True,
        )

        return GradientMaskExtensionOutput(
            guidance_module=GuidanceField(
                    guidance_name="mask_guidance",
                    priority=self.priority,
                    extension_kwargs=kwargs,
            ),
            expanded_mask_area=ImageField(
                image_name=expanded_image_dto.image_name
            )
        )
