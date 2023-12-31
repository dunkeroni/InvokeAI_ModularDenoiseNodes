from invokeai.app.invocations.latent import DenoiseLatentsInvocation
from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline

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

import math
from typing import Callable, List, Optional, Union, Any

import torch
import torch.nn.functional as F
import random
from invokeai.backend.ip_adapter.unet_patcher import UNetPatcher
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData

from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.t2i_adapter import T2IAdapterField

from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    IPAdapterData,
    T2IAdapterData
)
from invokeai.app.invocations.primitives import IntegerOutput, LatentsField, ImageField, ImageOutput
from pydantic import BaseModel, Field

from .modular_decorators import get_noise_prediction_module

import matplotlib.pyplot as plt
from PIL import Image
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecordChanges, ResourceOrigin
from diffusers.schedulers.scheduling_utils import SchedulerMixin
import numpy as np

import inspect #TODO: get rid of this garbage

@invocation("analyze_latents", title="Analyze Latents", tags=["analyze", "latents"], category="modular", version="1.0.0")
class AnalyzeLatentsInvocation(BaseInvocation):
    """ Create an image of a histogram of the latents with averages marked """
    latents: LatentsField = InputField(
        default=None, input=Input.Connection
    )
    bins: int = InputField(
        default=100,
        description="Number of bins to use in the histogram",
        title="Bins",
    )
    start_range: float = InputField(
        default=-4,
        input=Input.Direct,
        description="Start of the range to use in the histogram",
        title="Start Range",
    )
    end_range: float = InputField(
        default=4,
        input=Input.Direct,
        description="End of the range to use in the histogram",
        title="End Range",
    )
    image_title: str = InputField(
        default="Latent Histogram",
        input=Input.Direct,
        description="Title of the image",
        title="Image Title",
    )
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.services.latents.get(self.latents.latents_name)
        latents = latents.detach().cpu().numpy()
        #split individual channels
        L0 = latents[0,0,:,:]
        L1 = latents[0,1,:,:]
        L2 = latents[0,2,:,:]
        L3 = latents[0,3,:,:]

        #create histogram
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].hist(L0.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
        axs[0, 0].set_title('L0')
        axs[0, 1].hist(L1.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
        axs[0, 1].set_title('L1')
        axs[1, 0].hist(L2.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
        axs[1, 0].set_title('L2')
        axs[1, 1].hist(L3.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
        axs[1, 1].set_title('L3')

        #add title
        fig.suptitle(self.image_title)

        plt.tight_layout()  # Adjust subplot spacing

        #add average lines
        axs[0, 0].axvline(x=L0.mean(), color='r', linestyle='dashed', linewidth=1)
        axs[0, 1].axvline(x=L1.mean(), color='r', linestyle='dashed', linewidth=1)
        axs[1, 0].axvline(x=L2.mean(), color='r', linestyle='dashed', linewidth=1)
        axs[1, 1].axvline(x=L3.mean(), color='r', linestyle='dashed', linewidth=1)

        #conver to PIL image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)
        img = Image.fromarray(buf)

        #return image
        image_dto = context.services.images.create(
            image=img,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=img.width,
            height=img.height,
        )


class ModuleData(BaseModel):
    name: str = Field(description="Name of the module")
    module_type: str = Field(description="Type of module. Not yet used, may be in future")
    module: str = Field(description="Name of the module function")
    module_kwargs: dict | None = Field(description="Keyword arguments to pass to the module function")


@invocation_output("module_data_output")
class ModuleDataOutput(BaseInvocationOutput):
    module_data_output: ModuleData | None = OutputField(
        title="Module Data Output",
        description="Information for calling the module in denoise latents step()"
    )


#INHERIT to add custom module input
@invocation(
    "modular_denoise_latents",
    title="Modular Denoise Latents",
    tags=["modular", "generate", "denoise", "latents"],
    category="modular",
    version="1.4.0",
)
class Modular_DenoiseLatentsInvocation(DenoiseLatentsInvocation):
    module: Optional[ModuleData] = InputField(
        default=None,
        description="Information to override the default unet_step functions",
        title="Custom Modules",
        input=Input.Connection,
    )

    # OVERRIDE to use Modular_StableDiffusionGeneratorPipeline
    def create_pipeline(
        self,
        unet,
        scheduler,
    ) -> StableDiffusionGeneratorPipeline:
        class FakeVae:
            class FakeVaeConfig:
                def __init__(self):
                    self.block_out_channels = [0]

            def __init__(self):
                self.config = FakeVae.FakeVaeConfig()
        
        """ NOTE: This is gross and bad, and I feel bad for using it. HOWEVER...
            The modules might have data inputs that cannot be serialized (latents) so must be retrieved through the context.
            The DenoiseLatentsInvocation only has its context in the invoke() function.
            I don't want to maintain a custom override for invoke() on top of everything else here for compatibility reasons.
            So, I'm using inspect.stack() to get the context from the calling function [which is invoke(self, context)].

            If this ever gets merged into the main repo, just modify the create_pipeline() function to take a context argument.
        """
        for f in inspect.stack():
            if "context" in f[0].f_locals:
                context = f[0].f_locals["context"]

        return Modular_StableDiffusionGeneratorPipeline(
            vae=FakeVae(),  # TODO: oh...
            text_encoder=None,
            tokenizer=None,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            custom_module_data=self.module, #NEW field for custom module in pipeline
            context=context, #NEW for when modules need to load external data
        )



class Modular_StableDiffusionGeneratorPipeline(StableDiffusionGeneratorPipeline):

    def __init__(self, *args, **kwargs):
        self.custom_module_data = kwargs.pop("custom_module_data", None)
        self.persistent_data = {} # for modules storing data between steps
        self.context: InvocationContext = kwargs.pop("context", None)
        super().__init__(*args, **kwargs)


    def check_persistent_data(self, id: str, key: str):
        """Check if persistent data entry exists for the given id and key. Create None entry if not."""
        if id not in self.persistent_data:
            self.persistent_data[id] = {}
        if key not in self.persistent_data[id]:
            self.persistent_data[id][key] = None
            return None
        return self.persistent_data[id][key]
    

    def set_persistent_data(self, id: str, key: str, value: Any):
        """Set persistent data entry for the given id and key."""
        if id not in self.persistent_data:
            self.persistent_data[id] = {}
        self.persistent_data[id][key] = value
    

    def get_t2i_intrablock(self, t2i_adapter_data: list[T2IAdapterData], step_index, total_step_count):
        # Broken out of main step() function for easier calling later
        down_intrablock_additional_residuals = None
        if t2i_adapter_data is not None:
            accum_adapter_state = None
            for single_t2i_adapter_data in t2i_adapter_data:
                # Determine the T2I-Adapter weights for the current denoising step.
                first_t2i_adapter_step = math.floor(single_t2i_adapter_data.begin_step_percent * total_step_count)
                last_t2i_adapter_step = math.ceil(single_t2i_adapter_data.end_step_percent * total_step_count)
                t2i_adapter_weight = (
                    single_t2i_adapter_data.weight[step_index]
                    if isinstance(single_t2i_adapter_data.weight, list)
                    else single_t2i_adapter_data.weight
                )
                if step_index < first_t2i_adapter_step or step_index > last_t2i_adapter_step:
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

            # down_block_additional_residuals = accum_adapter_state
            down_intrablock_additional_residuals = accum_adapter_state
        
        return down_intrablock_additional_residuals

    #OVERRIDE to use custom_substep
    def step(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        conditioning_data: ConditioningData,
        step_index: int,
        total_step_count: int,
        additional_guidance: List[Callable] = None,
        control_data: List[ControlNetData] = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
        ip_adapter_unet_patcher: Optional[UNetPatcher] = None,
    ):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]
        if additional_guidance is None:
            additional_guidance = []

        # TODO: should this scaling happen here or inside self._unet_forward?
        #     i.e. before or after passing it to InvokeAIDiffuserComponent
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)

        # handle IP-Adapter OUTSIDE of custom modules
        if self.use_ip_adapter and ip_adapter_data is not None:  # somewhat redundant but logic is clearer
            for i, single_ip_adapter_data in enumerate(ip_adapter_data):
                first_adapter_step = math.floor(single_ip_adapter_data.begin_step_percent * total_step_count)
                last_adapter_step = math.ceil(single_ip_adapter_data.end_step_percent * total_step_count)
                weight = (
                    single_ip_adapter_data.weight[step_index]
                    if isinstance(single_ip_adapter_data.weight, List)
                    else single_ip_adapter_data.weight
                )
                if step_index >= first_adapter_step and step_index <= last_adapter_step:
                    # Only apply this IP-Adapter if the current step is within the IP-Adapter's begin/end step range.
                    ip_adapter_unet_patcher.set_scale(i, weight)
                else:
                    # Otherwise, set the IP-Adapter's scale to 0, so it has no effect.
                    ip_adapter_unet_patcher.set_scale(i, 0.0)


        custom_module_data: ModuleData = self.custom_module_data
        # default to standard diffusion pipeline result
        if custom_module_data is None:
            module_func: Callable = get_noise_prediction_module("standard_unet_step_module")
            module_kwargs = {}
        else:
            # invoke custom module
            module_func: Callable = get_noise_prediction_module(custom_module_data.module)
            module_kwargs = custom_module_data.module_kwargs
        ################################ CUSTOM SUBSTEP  ################################
        noise_pred, original_latents = module_func(
            self=self,
            latents = latents,
            sample=latent_model_input,
            t=t,  # TODO: debug how handled batched and non batched timesteps
            step_index=step_index,
            total_step_count=total_step_count,
            conditioning_data=conditioning_data,
            module_kwargs = module_kwargs,
            control_data=control_data, # passed down for tiling nodes to recalculate control blocks
            t2i_adapter_data=t2i_adapter_data, # passed down for tiling nodes to recalculate t2i blocks
        )
        ################################################################################

        # compute the previous noisy sample x_t -> x_t-1
        # here latents is substituted for the returned original_latents because the custom modules may have changed them
        step_output = self.scheduler.step(noise_pred, timestep, original_latents, **conditioning_data.scheduler_args)

        # TODO: issue to diffusers?
        # undo internal counter increment done by scheduler.step, so timestep can be resolved as before call
        # this needed to be able call scheduler.add_noise with current timestep
        if self.scheduler.order == 2:
            self.scheduler._index_counter[timestep.item()] -= 1

        # TODO: this additional_guidance extension point feels redundant with InvokeAIDiffusionComponent.
        #    But the way things are now, scheduler runs _after_ that, so there was
        #    no way to use it to apply an operation that happens after the last scheduler.step.
        for guidance in additional_guidance:
            step_output = guidance(step_output, timestep, conditioning_data)

        # restore internal counter
        if self.scheduler.order == 2:
            self.scheduler._index_counter[timestep.item()] += 1

        return step_output