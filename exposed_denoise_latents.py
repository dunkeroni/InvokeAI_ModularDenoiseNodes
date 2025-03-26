from contextlib import ExitStack
from typing import Type, Any, Optional, Callable, Union, List

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from invokeai.invocation_api import (
    invocation,
    invocation_output,
    BaseInvocationOutput,
    Input,
    InputField,
    OutputField,
    LatentsOutput,
    InvocationContext
)
from invokeai.app.invocations.fields import Field
from invokeai.backend.model_manager import ModelVariantType
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.stable_diffusion import PipelineIntermediateState
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext, DenoiseInputs

from invokeai.backend.stable_diffusion.diffusion.custom_atttention import CustomAttnProcessor2_0
from invokeai.backend.stable_diffusion.diffusion_backend import StableDiffusionBackend
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.freeu import FreeUExt
from invokeai.backend.stable_diffusion.extensions.inpaint import InpaintExt
from invokeai.backend.stable_diffusion.extensions.inpaint_model import InpaintModelExt
from invokeai.backend.stable_diffusion.extensions.lora import LoRAExt
from invokeai.backend.stable_diffusion.extensions.preview import PreviewExt
from invokeai.backend.stable_diffusion.extensions.rescale_cfg import RescaleCFGExt
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.stable_diffusion.extensions.t2i_adapter import T2IAdapterExt
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.schedulers import SCHEDULER_MAP
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.silence_warnings import SilenceWarnings


# imports purely for local implementation
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation
from invokeai.app.invocations.denoise_latents import get_scheduler
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.backend.util.logging import info, warning, error
from pydantic import BaseModel

from .extension_classes import SD12X_EXTENSIONS, GuidanceField, base_guidance_extension



#Current one in main is broken, so use this for now
class PreviewExtFIX(PreviewExt):
    def __init__(self, callback: Callable[[PipelineIntermediateState], None]):
        super().__init__(callback=callback)

    # do last so that all other changes shown
    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP, order=1000)
    def initial_preview(self, ctx: DenoiseContext):
        self.callback(
            PipelineIntermediateState(
                step=0, #-1 in main causes an error
                order=ctx.scheduler.order,
                total_steps=len(ctx.inputs.timesteps),
                timestep=int(ctx.scheduler.config.num_train_timesteps),  # TODO: is there any code which uses it?
                latents=ctx.latents,
            )
        )


@invocation(
    "exposed_denoise_latents",
    title="Exposed Denoise Latents",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="2.0.1",
)
class ExposedDenoiseLatentsInvocation(DenoiseLatentsInvocation):
    """includes all of the inputs and methods of the parent class"""

    guidance_extensions: Optional[Union[GuidanceField, List[GuidanceField]]] = InputField(
        default=None,
        description="Guidance information for extensions in the denoising process.",
        input=Input.Connection,
        ui_order=10,
    )

    @torch.no_grad()
    @SilenceWarnings()  # This quenches the NSFW nag from diffusers.
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        ext_manager = ExtensionsManager(is_canceled=context.util.is_canceled)

        device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_torch_dtype()

        seed, noise, latents = self.prepare_noise_and_latents(context, self.noise, self.latents)
        _, _, latent_height, latent_width = latents.shape

        conditioning_data = self.get_conditioning_data(
            context=context,
            positive_conditioning_field=self.positive_conditioning,
            negative_conditioning_field=self.negative_conditioning,
            cfg_scale=self.cfg_scale,
            steps=self.steps,
            latent_height=latent_height,
            latent_width=latent_width,
            device=device,
            dtype=dtype,
            # TODO: old backend, remove
            cfg_rescale_multiplier=self.cfg_rescale_multiplier,
        )

        # get the unet's config so that we can pass the base to sd_step_callback()
        unet_config = context.models.get_config(self.unet.unet.key)

        scheduler = get_scheduler(
            context=context,
            scheduler_info=self.unet.scheduler,
            scheduler_name=self.scheduler,
            seed=seed,
            unet_config=unet_config,
        )

        timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(
            scheduler,
            seed=seed,
            device=device,
            steps=self.steps,
            denoising_start=self.denoising_start,
            denoising_end=self.denoising_end,
        )

        # user extensions
        if self.guidance_extensions:
            if not isinstance(self.guidance_extensions, list):
                self.guidance_extensions = [self.guidance_extensions]

            for guidance in self.guidance_extensions:
                if guidance.guidance_name in SD12X_EXTENSIONS:
                    ext_cls = SD12X_EXTENSIONS[guidance.guidance_name]
                    ext = ext_cls(context=context, **guidance.extension_kwargs) #context required in case extension needs to load data on init
                    ext_manager.add_extension(ext)
                else:
                    raise ValueError(f"Extension {guidance.guidance_name} not found")

        # get the unet's config so that we can pass the base to sd_step_callback()
        unet_config = context.models.get_config(self.unet.unet.key)

        ### preview
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, unet_config.base)

        ext_manager.add_extension(PreviewExtFIX(step_callback))

        ### cfg rescale
        if self.cfg_rescale_multiplier > 0:
            ext_manager.add_extension(RescaleCFGExt(self.cfg_rescale_multiplier))

        ### freeu
        if self.unet.freeu_config:
            ext_manager.add_extension(FreeUExt(self.unet.freeu_config))

        ### lora
        if self.unet.loras:
            for lora_field in self.unet.loras:
                ext_manager.add_extension(
                    LoRAExt(
                        node_context=context,
                        model_id=lora_field.lora,
                        weight=lora_field.weight,
                    )
                )
        ### seamless
        if self.unet.seamless_axes:
            ext_manager.add_extension(SeamlessExt(self.unet.seamless_axes))

        ### inpaint
        mask, masked_latents, is_gradient_mask = self.prep_inpaint_mask(context, latents)
        # NOTE: We used to identify inpainting models by inpecting the shape of the loaded UNet model weights. Now we
        # use the ModelVariantType config. During testing, there was a report of a user with models that had an
        # incorrect ModelVariantType value. Re-installing the model fixed the issue. If this issue turns out to be
        # prevalent, we will have to revisit how we initialize the inpainting extensions.
        if unet_config.variant == ModelVariantType.Inpaint:
            ext_manager.add_extension(InpaintModelExt(mask, masked_latents, is_gradient_mask))
        elif mask is not None:
            ext_manager.add_extension(InpaintExt(mask, is_gradient_mask))

        # Initialize context for modular denoise
        latents = latents.to(device=device, dtype=dtype)
        if noise is not None:
            noise = noise.to(device=device, dtype=dtype)
        denoise_ctx = DenoiseContext(
            inputs=DenoiseInputs(
                orig_latents=latents,
                timesteps=timesteps,
                init_timestep=init_timestep,
                noise=noise,
                seed=seed,
                scheduler_step_kwargs=scheduler_step_kwargs,
                conditioning_data=conditioning_data,
                attention_processor_cls=CustomAttnProcessor2_0,
            ),
            unet=None,
            scheduler=scheduler,
        )

        # context for loading additional models
        with ExitStack() as exit_stack:
            # later should be smth like:
            # for extension_field in self.extensions:
            #    ext = extension_field.to_extension(exit_stack, context, ext_manager)
            #    ext_manager.add_extension(ext)
            self.parse_controlnet_field(exit_stack, context, self.control, ext_manager)
            self.parse_t2i_adapter_field(exit_stack, context, self.t2i_adapter, ext_manager)

            # ext: t2i/ip adapter
            ext_manager.run_callback(ExtensionCallbackType.SETUP, denoise_ctx)

            unet_info = context.models.load(self.unet.unet)
            assert isinstance(unet_info.model, UNet2DConditionModel)
            with (
                unet_info.model_on_device() as (cached_weights, unet),
                ModelPatcher.patch_unet_attention_processor(unet, denoise_ctx.inputs.attention_processor_cls),
                # ext: controlnet
                ext_manager.patch_extensions(denoise_ctx),
                # ext: freeu, seamless, ip adapter, lora
                ext_manager.patch_unet(unet, cached_weights),
            ):
                sd_backend = StableDiffusionBackend(unet, scheduler)
                denoise_ctx.unet = unet
                denoise_ctx.sd_backend = sd_backend # required for forced calls from extensions. Can this be done another way?
                result_latents = sd_backend.latents_from_embeddings(denoise_ctx, ext_manager)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        result_latents = result_latents.detach().to("cpu")
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)