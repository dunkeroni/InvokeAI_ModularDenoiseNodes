from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.invocations.fields import (
    InputField,
    LatentsField,
    ImageField,
)
from invokeai.invocation_api import (
    ImageOutput,
)
import torch
from .extension_classes import GuidanceField, base_guidance_extension, GuidanceDataOutput
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.util.logging import info, warning, error
import random
import einops
from PIL import Image
import numpy as np

@base_guidance_extension("FAM_FM")
class FAM_FM_Guidance(ExtensionBase):
    def __init__(
        self,
        context: InvocationContext,
        c: float,
        latent_image_name: str,
    ):
        self.c = c
        self.initial_latents = context.tensors.load(latent_image_name)
        self.noise = torch.randn(
            self.initial_latents.shape,
            dtype=torch.float32,
            device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(random.randint(0, 2 ** 32 - 1)),
        ).to(device=self.initial_latents.device, dtype=self.initial_latents.dtype)
        super().__init__()
    
    @callback(ExtensionCallbackType.PRE_STEP)
    @torch.no_grad()
    def pre_step(self, ctx: DenoiseContext):
        t = ctx.timestep
        if t.dim() == 0:
            # some schedulers expect t to be one-dimensional.
            # TODO: file diffusers bug about inconsistency?
            t = einops.repeat(t, "-> batch", batch=ctx.latents.size(0))
        
        latents = ctx.latents.clone().double()
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents, s=None, dim=(-2, -1), norm="ortho"))
        skip_residual = ctx.scheduler.add_noise(self.initial_latents, self.noise.to(self.initial_latents.device), t).double()
        skip_residual_fft = torch.fft.fftshift(torch.fft.fft2(skip_residual, s=None, dim=(-2, -1), norm="ortho").to(ctx.latents.device))
        K_t = torch.ones_like(self.initial_latents).to(ctx.latents.device)
        
        rho = ctx.timestep.item() / ctx.scheduler.config.num_train_timesteps
        print(f"rho: {rho}")
        h_i = self.initial_latents.shape[-2]
        w_i = self.initial_latents.shape[-1]
        tau_h = h_i * self.c * (1 - rho)
        tau_w = w_i * self.c * (1 - rho)
        print(f"tau_h: {tau_h}, tau_w: {tau_w}")
        print(f"h_i: {h_i}, w_i: {w_i}")

        h_d = latents.shape[-2]
        w_d = latents.shape[-1]

        # create a high-pass filter on the shifted domain
        # in horizontal dimension: K_t = rho if |X - Xc| < tau_w/2 else 1
        # in vertical dimension: K_t = rho if |Y - Yc| < tau_h/2 else 1
        K_t[:, :, int((h_i // 2) - (tau_h // 2)): int((h_i // 2) + (tau_h // 2)), int((w_i // 2) - (tau_w // 2)): int((w_i // 2) + (tau_w // 2))] = 1 - rho

        lf_part = skip_residual_fft * (1 - K_t)

        # pad the low frequency part equally in both directions with zeros
        pad_h = h_d - h_i
        pad_w = w_d - w_i
        lf_part = torch.nn.functional.pad(lf_part, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0)
        K_t_padded = torch.nn.functional.pad(K_t, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=1)

        #combine the low frequency components of the skip residual with the high frequency components of the latent image
        latents_fft = latents_fft * K_t_padded + lf_part

        #invert the FFT to get the new latent image
        ctx.latents = torch.fft.ifft2(torch.fft.ifftshift(latents_fft), s=None, dim=(-2, -1), norm="ortho").real.half()


@invocation(
    "frequency_modulation_extInvocation",
    title="I2I Preservation (FM) [Extension]",
    tags=["FAM", "frequency", "modulation", "extension"],
    category="latents",
    version="1.0.3",
)
class FAM_FM_ExtensionInvocation(BaseInvocation):
    """Preserves low frequency features from an input image."""
    c: float = InputField(
        title="c",
        description="'c' value for the FAM extension. Affects scaling of the cutoff frequency per step.",
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    latent_image: LatentsField = InputField(
        title="Latent Image",
        description="Latent image to be targeted.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:
        kwargs = {
            "c": self.c,
            "latent_image_name": self.latent_image.latents_name,
        }
        return GuidanceDataOutput(
            guidance_data_output=GuidanceField(
                guidance_name="FAM_FM",
                extension_kwargs=kwargs
            )
        )

