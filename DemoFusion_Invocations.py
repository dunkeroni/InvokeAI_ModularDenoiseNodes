from contextlib import ExitStack
from functools import singledispatchmethod
from typing import List, Literal, Optional, Union

import einops
import numpy as np
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.adapter import T2IAdapter
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.schedulers import DPMSolverSDEScheduler
from diffusers.schedulers import SchedulerMixin as Scheduler
from pydantic import field_validator
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.primitives import (
    DenoiseMaskField,
    DenoiseMaskOutput,
    ImageField,
    ImageOutput,
    LatentsField,
    LatentsOutput,
    build_latents_output,
)
from invokeai.app.invocations.t2i_adapter import T2IAdapterField
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.shared.fields import FieldDescriptions
from invokeai.app.util.controlnet_utils import prepare_control_image
from invokeai.app.util.step_callback import stable_diffusion_step_callback
from invokeai.backend.ip_adapter.ip_adapter import IPAdapter, IPAdapterPlus
from invokeai.backend.model_management.models import ModelType, SilenceWarnings
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData, IPAdapterConditioningInfo

from invokeai.backend.model_management.lora import ModelPatcher
from invokeai.backend.model_management.models import BaseModelType
from invokeai.backend.model_management.seamless import set_seamless
from invokeai.backend.stable_diffusion import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    IPAdapterData,
    StableDiffusionGeneratorPipeline,
    T2IAdapterData,
    image_resized_to_grid_as_tensor,
)
from invokeai.backend.stable_diffusion.diffusion.shared_invokeai_diffusion import PostprocessingSettings
from invokeai.backend.stable_diffusion.schedulers import SCHEDULER_MAP
from invokeai.backend.util.devices import choose_precision, choose_torch_device
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    UIType,
    WithMetadata,
    WithWorkflow,
    invocation,
    invocation_output,
)

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    ExtraConditioningInfo,
    SDXLConditioningInfo,
)

from invokeai.app.invocations.compel import ConditioningField
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.model import ModelInfo, UNetField, VaeField
from invokeai.app.invocations.model import MainModelField

from .DemoFusion_core import DemoFusionSDXLPipeline
from invokeai.backend.util.devices import choose_torch_device

if choose_torch_device() == torch.device("mps"):
    from torch import mps

"""
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline

model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Envision a portrait of an elderly woman, her face a canvas of time, framed by a headscarf with muted tones of rust and cream. Her eyes, blue like faded denim. Her attire, simple yet dignified."
negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

images = pipe(prompt, negative_prompt=negative_prompt,
              height=3072, width=3072, view_batch_size=16, stride=64,
              num_inference_steps=50, guidance_scale=7.5,
              cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
              multi_decoder=True, show_image=True
             )
"""


def get_scheduler(
        context: InvocationContext,
        scheduler_info: ModelInfo,
        scheduler_name: str,
        seed: int,
    ) -> Scheduler:
    scheduler_class, scheduler_extra_config = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP["ddim"])
    orig_scheduler_info = context.services.model_manager.get_model(
        **scheduler_info.model_dump(),
        context=context,
    )
    with orig_scheduler_info as orig_scheduler:
        scheduler_config = orig_scheduler.config

    if "_backup" in scheduler_config:
        scheduler_config = scheduler_config["_backup"]
    scheduler_config = {
        **scheduler_config,
        **scheduler_extra_config,
        "_backup": scheduler_config,
    }

    # make dpmpp_sde reproducable(seed can be passed only in initializer)
    if scheduler_class is DPMSolverSDEScheduler:
        scheduler_config["noise_sampler_seed"] = seed

    scheduler = scheduler_class.from_config(scheduler_config)

    # hack copied over from generate.py
    if not hasattr(scheduler, "uses_inpainting_model"):
        scheduler.uses_inpainting_model = lambda: False
    return scheduler

SAMPLER_NAME_VALUES = Literal[tuple(SCHEDULER_MAP.keys())]

@invocation(
    "demofusion_aio",
    title="DemoFusion (All-in-One)",
    tags=["demofusion", "generate"],
    category="demofusion",
    version="1.0.0",
)
class DemoFusionAllInOneInvocation(BaseInvocation):
    unet: UNetField = InputField(
        title="UNet",
        description="The UNet model.",
    )
    vae: VaeField = InputField(
        title="VAE",
        description="The VAE model.",
    )
    seed: int = InputField(
        title="Seed",
        description="The seed.",
        default=0,
    )
    scheduler: SAMPLER_NAME_VALUES = InputField(
        default="euler",
        description=FieldDescriptions.scheduler,
        ui_type=UIType.Scheduler,
    )
    positive_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection, ui_order=0
    )
    negative_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection, ui_order=1
    )
    height: int = InputField(
        title="Height",
        description="The height of the generated image.",
        multiple_of=1024,
        default=3072,
    )
    width: int = InputField(
        title="Width",
        description="The width of the generated image.",
        multiple_of=1024,
        default=3072,
    )
    view_batch_size: int = InputField(
        title="View Batch Size",
        description="The batch size of the generated image.",
        default=16,
    )
    stride: int = InputField(
        title="Stride",
        description="The stride of the generated image.",
        default=64,
    )
    num_inference_steps: int = InputField(
        title="Number of Inference Steps",
        description="The number of inference steps.",
        default=50,
    )
    guidance_scale: float = InputField(
        title="Guidance Scale",
        description="The guidance scale.",
        default=5,
    )
    cosine_scale_1: float = InputField(
        title="Cosine Scale 1",
        description="The cosine scale 1.",
        default=3,
    )
    cosine_scale_2: float = InputField(
        title="Cosine Scale 2",
        description="The cosine scale 2.",
        default=1,
    )
    cosine_scale_3: float = InputField(
        title="Cosine Scale 3",
        description="The cosine scale 3.",
        default=1,
    )
    sigma: float = InputField(
        title="Sigma",
        description="The sigma.",
        default=0.8,
    )
    multi_decoder: bool = InputField(
        title="Multi Decoder",
        description="The multi decoder.",
        default=True,
    )

    def dispatch_progress(
        self,
        context: InvocationContext,
        source_node_id: str,
        intermediate_state: PipelineIntermediateState,
        base_model: BaseModelType,
    ) -> None:
        stable_diffusion_step_callback(
            context=context,
            intermediate_state=intermediate_state,
            node=self.model_dump(),
            source_node_id=source_node_id,
            base_model=base_model,
        )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        
        # def __init__(
        #     self,
        #     vae: AutoencoderKL,
        #     text_encoder: CLIPTextModel,
        #     text_encoder_2: CLIPTextModelWithProjection,
        #     tokenizer: CLIPTokenizer,
        #     tokenizer_2: CLIPTokenizer,
        #     unet: UNet2DConditionModel,
        #     scheduler: KarrasDiffusionSchedulers,
        #     force_zeros_for_empty_prompt: bool = True,
        #     add_watermarker: Optional[bool] = None,
        # ):

        def step_callback(state: PipelineIntermediateState):
                self.dispatch_progress(context, self.id, state, self.unet.unet.base_model)

        device = choose_torch_device()

        unet_info: ModelInfo = context.services.model_manager.get_model(
                **self.unet.unet.model_dump(),
                context=context,
            )
        
        vae_info = context.services.model_manager.get_model(
                **self.vae.vae.model_dump(),
                context=context,
            )

        scheduler = get_scheduler(
                    context=context,
                    scheduler_info=self.unet.scheduler,
                    scheduler_name=self.scheduler,
                    seed=self.seed,
                )

        positive_conditioning: SDXLConditioningInfo = context.services.latents.get(self.positive_conditioning.conditioning_name).conditionings[0]
        positive_embeds = positive_conditioning.embeds.to(device)
        positive_pooled_embeds = positive_conditioning.pooled_embeds.to(device)

        negative_conditioning: SDXLConditioningInfo = context.services.latents.get(self.negative_conditioning.conditioning_name).conditionings[0]
        negative_embeds = negative_conditioning.embeds.to(device)
        negative_pooled_embeds = negative_conditioning.pooled_embeds.to(device)


        pipe = DemoFusionSDXLPipeline(
            vae = vae_info.context.model.to(device),
            positive_embeds = positive_embeds,
            positive_pooled_embeds = positive_pooled_embeds,
            negative_embeds = negative_embeds,
            negative_pooled_embeds = negative_pooled_embeds,
            unet = unet_info.context.model.to(device),
            scheduler = scheduler,
            force_zeros_for_empty_prompt = True,
            add_watermarker = None,
        )
        result_latents = pipe(
                    height=self.height, width=self.width, view_batch_size=self.view_batch_size, stride=self.stride,
                    num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale,
                    cosine_scale_1=self.cosine_scale_1, cosine_scale_2=self.cosine_scale_2, cosine_scale_3=self.cosine_scale_3, sigma=self.sigma,
                    multi_decoder=self.multi_decoder, show_image=False, callback=step_callback,
                    )
        
        result_latents = result_latents.to("cpu")
        torch.cuda.empty_cache()
        if choose_torch_device() == torch.device("mps"):
            mps.empty_cache()
        
        name = f"{context.graph_execution_state_id}__{self.id}"
        context.services.latents.save(name, result_latents)
        return build_latents_output(latents_name=name, latents=result_latents, seed=self.seed)
        
        # custom_pipeline = DemoFusionSDXLPipeline(
        #     vae = 
        # )