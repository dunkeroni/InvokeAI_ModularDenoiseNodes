
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
    invocation,
    invocation_output,
)

from invokeai.app.invocations.model import VaeField
from invokeai.app.invocations.latent import ImageToLatentsInvocation

if choose_torch_device() == torch.device("mps"):
    from torch import mps
DEFAULT_PRECISION = choose_precision(choose_torch_device())

SAMPLER_NAME_VALUES = Literal[tuple(SCHEDULER_MAP.keys())]

@invocation(
    "create_gradient_mask",
    title="Create Gradient Denoise Mask",
    tags=["mask", "denoise"],
    category="latents",
    version="1.0.0",
)
class CreateGradientMaskInvocation(BaseInvocation):
    """Creates mask for denoising model run."""

    vae: VaeField = InputField(description=FieldDescriptions.vae, input=Input.Connection, ui_order=0)
    image: Optional[ImageField] = InputField(default=None, description="Image which will be masked", ui_order=1)
    mask: ImageField = InputField(description="The mask to use when pasting", ui_order=2)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled, ui_order=3)
    fp32: bool = InputField(
        default=DEFAULT_PRECISION == "float32",
        description=FieldDescriptions.fp32,
        ui_order=4,
    )

    def prep_mask_tensor(self, mask_image):
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        # if shape is not None:
        #    mask_tensor = tv_resize(mask_tensor, shape, T.InterpolationMode.BILINEAR)
        return mask_tensor

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> DenoiseMaskOutput:
        if self.image is not None:
            image = context.services.images.get_pil_image(self.image.image_name)
            image = image_resized_to_grid_as_tensor(image.convert("RGB"))
            if image.dim() == 3:
                image = image.unsqueeze(0)
        else:
            image = None

        mask = self.prep_mask_tensor(
            context.services.images.get_pil_image(self.mask.image_name),
        )

        if image is not None:
            vae_info = context.services.model_manager.get_model(
                **self.vae.vae.model_dump(),
                context=context,
            )

            img_mask = tv_resize(mask, image.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
            masked_image = image * torch.where(img_mask < 0.5, 0.0, 1.0)
            # TODO:
            masked_latents = ImageToLatentsInvocation.vae_encode(vae_info, self.fp32, self.tiled, masked_image.clone())

            masked_latents_name = f"{context.graph_execution_state_id}__{self.id}_masked_latents"
            context.services.latents.save(masked_latents_name, masked_latents)
        else:
            masked_latents_name = None

        mask_name = f"{context.graph_execution_state_id}__{self.id}_mask"
        context.services.latents.save(mask_name, mask)

        return DenoiseMaskOutput(
            denoise_mask=DenoiseMaskField(
                mask_name=mask_name,
                masked_latents_name=masked_latents_name,
            ),
        )

@invocation(
    "extract_latents_mask",
    title="Extract Masked_Latents from Mask",
    tags=["mask", "denoise"],
    category="latents",
    version="1.0.0",
)
class ExtractLatentsMaskInvocation(BaseInvocation):
    """gets the masked_latents from the mask"""

    mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.mask,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        masked_latents = context.services.latents.get(self.mask.masked_latents_name)

        return build_latents_output(self.mask.masked_latents_name, masked_latents)

@invocation(
    "extract_mask",
    title="Extract Mask from DenoiseMaskField",
    tags=["mask", "denoise"],
    category="latents",
    version="1.0.0",
)
class ExtractMaskInvocation(BaseInvocation):
    """gets the mask from the DenoiseMaskField"""

    mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.mask,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        mask = context.services.latents.get(self.mask.mask_name)

        return build_latents_output(self.mask.mask_name, mask)