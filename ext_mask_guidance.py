from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X

import einops
import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torchvision.transforms.functional import resize as tv_resize

@guidance_extension_12X("mask_guidance")
class MaskGuidance(DenoiseExtensionSD12X):

    def __post_init__(self, mask_name: str, masked_latents_name: str | None, gradient_mask: bool):
        """Align internal data and create noise if necessary"""
        context = self.input_data.context
        if self.input_data.latents is not None:
            self.orig_latents = context.tensors.load(self.input_data.latents.latents_name)
        else:
            raise ValueError("Latents input is required for the denoise mask extension")
        if self.input_data.noise is not None:
            self.noise = context.tensors.load(self.input_data.noise.latents_name)
        else:
            self.noise = torch.randn(
                self.orig_latents.shape,
                dtype=torch.float32,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(self.input_data.seed),
            ).to(device=self.orig_latents.device, dtype=self.orig_latents.dtype)

        self.mask: torch.Tensor = context.tensors.load(mask_name)
        self.masked_latents = None if masked_latents_name is None else context.tensors.load(masked_latents_name)
        self.scheduler: SchedulerMixin = self.input_data.scheduler
        self.gradient_mask: bool = gradient_mask
        self.unet_type: str = self.input_data.unet.unet.base
        self.inpaint_model = self.input_data.unet_model.conv_in.in_channels == 9
        self.seed: int = self.input_data.seed

        self.mask = tv_resize(self.mask, list(self.orig_latents.shape[-2:]))
        self.mask = self.mask.to(device=self.orig_latents.device, dtype=self.orig_latents.dtype)
    
    def list_modifies(self) -> dict[str, function]:
        return {
            "modify_data_before_scaling": self.modify_data_before_scaling,
            "modify_data_before_noise_prediction": self.modify_data_before_noise_prediction,
            "modify_result_before_callback": self.modify_result_before_callback,
            "modify_data_after_denoising": self.modify_data_after_denoising,
            }
    
    def list_swaps(self) -> dict[str, function]:
        return super().list_swaps()

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

    def modify_data_before_scaling(self, data: DenoiseLatentsData, t: torch.Tensor) -> DenoiseLatentsData:
        """Replace unmasked region with original latents. Called before the scheduler scales the latent values."""
        if self.inpaint_model:
            return data # skip this stage

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
        masked_input = torch.lerp(latents, noised_latents, mask)

        data.latents = masked_input
        return data

    def shrink_mask(self, mask: torch.Tensor, n_operations: int) -> torch.Tensor:
        kernel = torch.ones(1, 1, 3, 3).to(device=mask.device, dtype=mask.dtype)
        for _ in range(n_operations):
            mask = torch.nn.functional.conv2d(mask, kernel, padding=1).clamp(0, 1)
        return mask

    def modify_data_before_noise_prediction(self, data: DenoiseLatentsData, t: torch.Tensor) -> DenoiseLatentsData:
        """Expand latents with information needed by inpaint model"""
        if not self.inpaint_model:
            return data # skip this stage

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
        return data

    def modify_result_before_callback(self, step_output, t) -> torch.Tensor:
        """Fix preview images to show the original image in the unmasked region"""
        if hasattr(step_output, "denoised"): #LCM Sampler
            prediction = step_output.denoised
        elif hasattr(step_output, "pred_original_sample"): #Samplers with final predictions
            prediction = step_output.pred_original_sample
        else: #all other samplers (no prediction available)
            prediction = step_output.prev_sample

        mask = self.mask_from_timestep(t)
        mask = einops.repeat(self.mask, "b c h w -> (repeat b) c h w", repeat=prediction.size(0))
        step_output.pred_original_sample = torch.lerp(prediction, self.orig_latents.to(dtype=prediction.dtype), mask.to(dtype=prediction.dtype))

        return step_output

    def modify_data_after_denoising(self, data: DenoiseLatentsData) -> DenoiseLatentsData:
        """Apply original unmasked to denoised latents"""
        if self.inpaint_model:
            if self.masked_latents is None:
                mask = self.shrink_mask(self.mask, 1)
            else:
                return data
        else:
            mask = self.mask_from_timestep(torch.Tensor([0]))
        latents = data.latents
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        latents = torch.lerp(latents, self.orig_latents.to(dtype=latents.dtype), mask.to(dtype=latents.dtype)).to(device=latents.device)

        data.latents = latents
        return data
