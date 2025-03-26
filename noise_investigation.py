from typing import Literal, Optional, Union, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    ImageField,
    LatentsField,
    Input,
    InputField,
    OutputField,
    UIType,
)
from invokeai.invocation_api import ImageOutput, LatentsOutput
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

@invocation(
    "noise_heatmap",
    title="Noise Progression Heatmap",
    tags=["latents", "noise", "heatmap"],
    category="latents",
    version="1.0.0",
)
class NoiseHeatmapInvocation(BaseInvocation):
    noise: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.noise,
        input=Input.Connection,
    )
    latents: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        """for each batch element in latents, generate a similarity heatmap comparing it to the noise"""
        latents = context.tensors.load(self.latents.latents_name)
        noise = context.tensors.load(self.noise.latents_name)
        batch_size = latents.shape[0]

        # Convert latents and noise to the shifted Fourier frequency domain
        latents = latents.to(torch.float32)
        noise = noise.to(torch.float32)
        latents_fft2 = torch.fft.fft2(latents, dim=(-2, -1))
        noise_fft2 = torch.fft.fft2(noise, dim=(-2, -1))

        # convert to shifted Fourier frequency domain
        latents_fft = torch.fft.fftshift(latents_fft2)
        noise_fft = torch.fft.fftshift(noise_fft2)

        # compute similarity heatmap
        heatmap_list = []
        heatmap_i_list = []
        for i in range(latents.shape[1]):
            heatmap = (latents_fft[0, i, :, :] - noise_fft[0, i, :, :]).real.abs()
            heatmap_i = (latents_fft[0, i, :, :] - noise_fft[0, i, :, :]).imag.abs()
            heatmap = torch.clamp(heatmap, -16, 16)
            heatmap_i = torch.clamp(heatmap_i, -16, 16)
            # normalize heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap_i = (heatmap_i - heatmap_i.min()) / (heatmap_i.max() - heatmap_i.min())
            # convert to a PIL image
            heatmap = heatmap.squeeze().cpu().numpy()
            heatmap_i = heatmap_i.squeeze().cpu().numpy()
            heatmap = np.transpose(heatmap, (1, 2, 0)) if heatmap.ndim == 3 else heatmap
            heatmap_i = np.transpose(heatmap_i, (1, 2, 0)) if heatmap_i.ndim == 3 else heatmap_i
            heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).convert("RGB")
            heatmap_i = Image.fromarray((heatmap_i * 255).astype(np.uint8)).convert("RGB")
            heatmap = heatmap.resize((256, 256), Image.Resampling.NEAREST)
            heatmap_i = heatmap_i.resize((256, 256), Image.Resampling.NEAREST)
            # Add red border
            heatmap = ImageOps.expand(heatmap, border=5, fill='red')
            heatmap_i = ImageOps.expand(heatmap_i, border=5, fill='green')
            heatmap_list.append(heatmap)
            heatmap_i_list.append(heatmap_i)

        # arrange heatmaps in a grid
        combined_heatmap = Image.new('RGB', (266 * 4, 266 * 2))
        for idx, heatmap in enumerate(heatmap_list):
            x = (idx % 2) * 266
            y = (idx // 2) * 266
            combined_heatmap.paste(heatmap, (x, y))

        for idx, heatmap_i in enumerate(heatmap_i_list):
            x = (idx % 2) * 266 + 532
            y = (idx // 2) * 266
            combined_heatmap.paste(heatmap_i, (x, y))

        image_dto = context.images.save(combined_heatmap)
        
        return ImageOutput.build(image_dto)

@invocation(
    "FourierLossCheck",
    title="Fourier Loss Check",
    tags=["latents", "noise", "heatmap"],
    category="latents",
    version="1.0.0",
)
class FourierLossCheckInvocation(BaseInvocation):
    """convert a latent to shifted fourer space and back before saving"""

    latents: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.tensors.load(self.latents.latents_name)

        # Convert latents to the shifted Fourier frequency domain
        latents = latents.to(torch.float32)
        latents_fft2 = torch.fft.fft2(latents, dim=(-2, -1))
        latents_fft = torch.fft.fftshift(latents_fft2)

        # convert back to spatial domain
        latents = torch.fft.ifftn(torch.fft.ifftshift(latents_fft), dim=(-2, -1)).real

        latents_name = context.tensors.save(latents)
        
        return LatentsOutput.build(latents_name, latents)
    
@invocation(
    "CopyFrequencyValues",
    title="Copy Frequency Values",
    tags=["latents", "noise"],
    category="latents",
    version="1.0.0",
)
class CopyFrequencyValuesInvocation(BaseInvocation):
    source_tensor: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.noise,
        input=Input.Connection,
    )
    target_tensor: LatentsField = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    cutoff: float = InputField(
        default=0.1,
        description="cutoff frequency",
        ge=0,
        le=1,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        source_tensor = context.tensors.load(self.source_tensor.latents_name)
        target_tensor = context.tensors.load(self.target_tensor.latents_name)

        # Convert latents to the shifted Fourier frequency domain
        source_tensor = source_tensor.to(torch.float32)
        target_tensor = target_tensor.to(torch.float32)
        source_fft2 = torch.fft.fft2(source_tensor, dim=(-2, -1), s = None, norm="ortho")
        target_fft2 = torch.fft.fft2(target_tensor, dim=(-2, -1), s = None, norm="ortho")

        # convert to shifted Fourier frequency domain
        source_fft = torch.fft.fftshift(source_fft2)
        target_fft = torch.fft.fftshift(target_fft2)

        # copy the highest cutoff % of frequency values from source to target
        center_w = source_fft.shape[-1] // 2
        center_h = source_fft.shape[-2] // 2
        cutoff_w = center_w - int(center_w * self.cutoff)
        cutoff_h = center_h - int(center_h * self.cutoff)
        target_fft[:, :, center_h - cutoff_h:center_h + cutoff_h, center_w - cutoff_w:center_w + cutoff_w] = source_fft[:, :, center_h - cutoff_h:center_h + cutoff_h, center_w - cutoff_w:center_w + cutoff_w]

        # convert back to spatial domain
        combined_tensor = torch.fft.ifftn(torch.fft.ifftshift(target_fft), dim=(-2, -1), norm="ortho").real.half()

        # wherever the combined_tensor is >4 or <-4, replace it with the target_tensor
        combined_tensor = torch.where((combined_tensor > 4) | (combined_tensor < -4), target_tensor, combined_tensor)

        latents_name = context.tensors.save(combined_tensor)
        
        return LatentsOutput.build(latents_name, combined_tensor)

from invokeai.app.invocations.denoise_latents import get_scheduler
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
import einops

@invocation(
    "ScheduledNoise",
    title="Scheduled Noise",
    tags=["latents", "noise"],
    category="latents",
    version="1.0.0",
)
class ScheduledNoiseInvocation(BaseInvocation):
    unet: UNetField = InputField(
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    scheduler: SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description=FieldDescriptions.scheduler,
        ui_type=UIType.Scheduler,
    )
    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
        ui_order=4,
    )
    noise: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.noise,
        input=Input.Connection,
    )
    denoise_start: float = InputField(
        default=0.5,
        description="Start of denoising",
        ge=0,
        le=1,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        unet_config = context.models.get_config(self.unet.unet.key)
        scheduler = get_scheduler(
            context=context,
            scheduler_info=self.unet.scheduler,
            scheduler_name=self.scheduler,
            seed=0,
            unet_config=unet_config
        )
        device = TorchDevice.choose_torch_device()
        timesteps, init_timestep, scheduler_step_kwargs = DenoiseLatentsInvocation.init_scheduler(
            scheduler,
            seed=0,
            device=device,
            steps=100,
            denoising_start=0,
            denoising_end=1,
        )

        latents = context.tensors.load(self.latents.latents_name)
        noise = context.tensors.load(self.noise.latents_name)

        timestep = torch.tensor(self.denoise_start * scheduler.config.num_train_timesteps)

        #find the closest value in timesteps to timestep
        timestep = timesteps[(timesteps - timestep).abs().argmin()]

        timestep = einops.repeat(timestep, "-> batch", batch=1)
        
        # apply noise
        latents = scheduler.add_noise(latents, noise, timestep)

        latents_name = context.tensors.save(latents)
        
        return LatentsOutput.build(latents_name, latents)