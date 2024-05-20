from __future__ import annotations
from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline, PipelineIntermediateState
from .denoise_latents_extensions import ExtensionHandlerSD12X, DenoiseLatentsData

import math
from contextlib import nullcontext
from typing import Any, Callable, List, Optional

import torch
from invokeai.backend.stable_diffusion.diffusion.unet_attention_patcher import UNetAttentionPatcher, UNetIPAdapterData
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    IPAdapterData,
    TextConditioningData,
)

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from invokeai.backend.stable_diffusion.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent

class ExtendableStableDiffusionGeneratorPipeline(StableDiffusionGeneratorPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Implementation note: This class started as a refactored copy of diffusers.StableDiffusionPipeline.
    Hopefully future versions of diffusers provide access to more of these functions so that we don't
    need to duplicate them here: https://github.com/huggingface/diffusers/issues/551#issuecomment-1281508384

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: Optional[StableDiffusionSafetyChecker],
        feature_extractor: Optional[CLIPFeatureExtractor],
        requires_safety_checker: bool = False,
        control_model: ControlNetModel = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

        self.invokeai_diffuser = ExtendableInvokeAIDiffuserComponent(self.unet, self._unet_forward)
        self.control_model = control_model
        self.use_ip_adapter = False

    def latents_from_embeddings(
        self,
        data: DenoiseLatentsData,
        extension_handler: ExtensionHandlerSD12X,
        callback: Callable[[PipelineIntermediateState], None] = None,
    ) -> torch.Tensor:
        #TODO Do we even need this function layer here? Could call directly to generate_latents_from_embeddings instead

        if data.init_timestep.shape[0] == 0:
            return data.latents

        batch_size = data.latents.shape[0]
        batched_t = data.init_timestep.expand(batch_size)

        if data.noise is not None:
            # latents = noise * self.scheduler.init_noise_sigma # it's like in t2l according to diffusers
            data.latents = self.scheduler.add_noise(data.latents, data.noise, batched_t)

        try:
            data.latents = self.generate_latents_from_embeddings(
                data = data,
                extension_handler = extension_handler,
                callback=callback,
            )
        finally:
            self.invokeai_diffuser.model_forward_callback = self._unet_forward

        extension_handler.call_modifiers("modify_data_after_denoising", data=data)
        return data.latents

    def generate_latents_from_embeddings(
        self,
        data: DenoiseLatentsData,
        extension_handler: ExtensionHandlerSD12X,
        callback: Callable[[PipelineIntermediateState], None] = None,
    ) -> torch.Tensor:
        self._adjust_memory_efficient_attention(data.latents)

        batch_size = data.latents.shape[0]

        if data.timesteps.shape[0] == 0:
            return data.latents

        use_ip_adapter = data.ip_adapter_data is not None
        use_regional_prompting = (
            data.conditioning_data.cond_regions is not None or data.conditioning_data.uncond_regions is not None
        )
        unet_attention_patcher = None
        self.use_ip_adapter = use_ip_adapter
        attn_ctx = nullcontext()

        if use_ip_adapter or use_regional_prompting:
            ip_adapters: Optional[List[UNetIPAdapterData]] = (
                [{"ip_adapter": ipa.ip_adapter_model, "target_blocks": ipa.target_blocks} for ipa in data.ip_adapter_data]
                if use_ip_adapter
                else None
            )
            unet_attention_patcher = UNetAttentionPatcher(ip_adapters)
            attn_ctx = unet_attention_patcher.apply_ip_adapter_attention(self.invokeai_diffuser.model)

        with attn_ctx:
            if callback is not None:
                callback(
                    PipelineIntermediateState(
                        step=-1,
                        order=self.scheduler.order,
                        total_steps=len(data.timesteps),
                        timestep=self.scheduler.config.num_train_timesteps,
                        latents=data.latents,
                    )
                )

            # print("timesteps:", timesteps)
            for i, t in enumerate(self.progress_bar(data.timesteps)):
                batched_t = t.expand(batch_size)
                data.step_index = i
                step_output = self.step(
                    batched_t,
                    data = data,
                    extension_handler = extension_handler,
                )
                data.latents = step_output.prev_sample

                extension_handler.call_modifiers(
                    "modify_result_before_callback",
                    step_output=step_output,
                    data=data, t=batched_t[0]
                )
                predicted_original = getattr(step_output, "pred_original_sample", None)

                if callback is not None:
                    callback(
                        PipelineIntermediateState(
                            step=i,
                            order=self.scheduler.order,
                            total_steps=len(data.timesteps),
                            timestep=int(t),
                            latents=data.latents,
                            predicted_original=predicted_original,
                        )
                    )

            return data.latents

    @torch.inference_mode()
    def step(
        self,
        t: torch.Tensor,
        data: DenoiseLatentsData,
        extension_handler: ExtensionHandlerSD12X,
    ):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]

        extension_handler.call_modifiers("modify_data_before_scaling", data=data, t=timestep)

        data.scaled_model_inputs = self.scheduler.scale_model_input(data.latents, timestep)

        # Handle ControlNet(s)
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if data.controlnet_data is not None:
            down_block_additional_residuals, mid_block_additional_residual = self.invokeai_diffuser.do_controlnet_step(
                control_data=data.controlnet_data,
                sample=data.scaled_model_inputs,
                timestep=timestep,
                step_index=data.step_index,
                total_step_count=len(data.timesteps),
                conditioning_data=data.conditioning_data,
            )

        # Handle T2I-Adapter(s)
        down_intrablock_additional_residuals = None
        if data.t2i_adapter_data is not None:
            accum_adapter_state = None
            for single_t2i_adapter_data in data.t2i_adapter_data:
                # Determine the T2I-Adapter weights for the current denoising step.
                first_t2i_adapter_step = math.floor(single_t2i_adapter_data.begin_step_percent * len(data.timesteps))
                last_t2i_adapter_step = math.ceil(single_t2i_adapter_data.end_step_percent * len(data.timesteps))
                t2i_adapter_weight = (
                    single_t2i_adapter_data.weight[data.step_index]
                    if isinstance(single_t2i_adapter_data.weight, list)
                    else single_t2i_adapter_data.weight
                )
                if data.step_index < first_t2i_adapter_step or data.step_index > last_t2i_adapter_step:
                    # If the current step is outside of the T2I-Adapter's begin/end step range, then set its weight to 0
                    # so it has no effect.
                    t2i_adapter_weight = 0.0

                # Apply the t2i_adapter_weight, and accumulate.
                if accum_adapter_state is None:
                    # Handle the first T2I-Adapter.
                    accum_adapter_state = [val * t2i_adapter_weight for val in single_t2i_adapter_data.adapter_state]
                else:
                    # Add to the previous adapter states.
                    for idx, value in enumerate(single_t2i_adapter_data.adapter_state):
                        accum_adapter_state[idx] += value * t2i_adapter_weight

            down_intrablock_additional_residuals = accum_adapter_state
        
        extension_handler.call_modifiers("modify_data_before_noise_prediction", data=data, t=timestep)

        uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
            sample=data.scaled_model_inputs,
            timestep=t,  # TODO: debug how handled batched and non batched timesteps
            step_index=data.step_index,
            total_step_count=len(data.timesteps),
            conditioning_data=data.conditioning_data,
            ip_adapter_data=data.ip_adapter_data,
            down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
            mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
        )

        guidance_scale = data.conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[data.step_index]

        noise_pred = extension_handler.call_swap(
            "combine_noise",
            default=self.invokeai_diffuser._combine,
            unconditioned_next_x=uc_noise_pred,
            conditioned_next_x=c_noise_pred,
            guidance_scale=guidance_scale
        )

        # compute the previous noisy sample x_t -> x_t-1
        step_output = self.scheduler.step(noise_pred, timestep, data.latents, **data.scheduler_step_kwargs)

        return step_output

    @staticmethod
    def _rescale_cfg(total_noise_pred, pos_noise_pred, multiplier=0.7):
        """Implementation of Algorithm 2 from https://arxiv.org/pdf/2305.08891.pdf."""
        ro_pos = torch.std(pos_noise_pred, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(total_noise_pred, dim=(1, 2, 3), keepdim=True)

        x_rescaled = total_noise_pred * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * total_noise_pred
        return x_final

    def _unet_forward(
        self,
        latents,
        t,
        text_embeddings,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        # First three args should be positional, not keywords, so torch hooks can see them.
        return self.unet(
            latents,
            t,
            text_embeddings,
            cross_attention_kwargs=cross_attention_kwargs,
            **kwargs,
        ).sample



class ExtendableInvokeAIDiffuserComponent(InvokeAIDiffuserComponent):

    def do_unet_step(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        conditioning_data: TextConditioningData,
        ip_adapter_data: Optional[list[IPAdapterData]],
        step_index: int,
        total_step_count: int,
        down_block_additional_residuals: Optional[torch.Tensor] = None,  # for ControlNet
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # for ControlNet
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,  # for T2I-Adapter
    ):
        if self.sequential_guidance:
            (
                unconditioned_next_x,
                conditioned_next_x,
            ) = self._apply_standard_conditioning_sequentially(
                x=sample,
                sigma=timestep,
                conditioning_data=conditioning_data,
                ip_adapter_data=ip_adapter_data,
                step_index=step_index,
                total_step_count=total_step_count,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            )
        else:
            (
                unconditioned_next_x,
                conditioned_next_x,
            ) = self._apply_standard_conditioning(
                x=sample,
                sigma=timestep,
                conditioning_data=conditioning_data,
                ip_adapter_data=ip_adapter_data,
                step_index=step_index,
                total_step_count=total_step_count,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            )

        return unconditioned_next_x, conditioned_next_x
