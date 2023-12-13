from invokeai.app.invocations.latent import DenoiseLatentsInvocation
from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline
from .DemoFusion_SDGP import DF_StableDiffusionGeneratorPipeline

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
from invokeai.app.invocations.primitives import IntegerOutput
from pydantic import BaseModel, Field

from .modular_noise_prediction import get_noise_prediction_module


class ModuleData(BaseModel):
    name: str = Field(description="Name of the module")
    module_type: str = Field(description="Type of module. Not yet used, may be in future")
    module: str = Field(description="Name of the module function")
    module_kwargs: dict | None = Field(description="Keyword arguments to pass to the module function")


@invocation_output("module_data_output")
class ModuleDataOutput(BaseInvocationOutput):
    module_data_output: ModuleData | None = OutputField(
        title="Module Data Output",
        description="Information for calling the module in denoise latents step()",
        ui_type=UIType.Any,
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
        ui_type=UIType.Any,
    )

    #control: Optional[Union[ControlField, list[ControlField]]] = None # remove from inputs
    #ip_adapter: Optional[Union[IPAdapterField, list[IPAdapterField]]] = None # remove from inputs
    #t2i_adapter: Optional[Union[T2IAdapterField, list[T2IAdapterField]]] = None # remove from inputs

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
        )



class Modular_StableDiffusionGeneratorPipeline(StableDiffusionGeneratorPipeline):

    def __init__(self, *args, **kwargs):
        self.custom_module_data = kwargs.pop("custom_module_data", None)
        super().__init__(*args, **kwargs)


    #NEW function, called in step() to replace previous self.invokeai_diffuser.do_unet_step
    def custom_substep(
        self,
        sample: torch.Tensor,
        t: torch.Tensor,
        conditioning_data,  # TODO: type
        step_index: int,
        total_step_count: int,
        custom_module_data: ModuleData,
        **kwargs,
    ) -> torch.Tensor:
        
        # default to standard diffusion pipeline result
        if custom_module_data is None:
            uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
                sample=sample,
                timestep=t,  # TODO: debug how handled batched and non batched timesteps
                step_index=step_index,
                total_step_count=total_step_count,
                conditioning_data=conditioning_data,
                # extra:
                #down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
                #mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
                #down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
                **kwargs
            )

            guidance_scale = conditioning_data.guidance_scale
            if isinstance(guidance_scale, list):
                guidance_scale = guidance_scale[step_index]

            return self.invokeai_diffuser._combine(
                uc_noise_pred,
                c_noise_pred,
                guidance_scale,
            )

        # invoke custom module
        module_func: Callable = get_noise_prediction_module(custom_module_data.module)

        return module_func(
            self=self,
            sample=sample,
            t=t,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            module_kwargs=custom_module_data.module_kwargs,
            **kwargs,
            ) # recursive case

        
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

        # handle IP-Adapter
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

        # Handle ControlNet(s) and T2I-Adapter(s)
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        down_intrablock_additional_residuals = None
        # if control_data is not None and t2i_adapter_data is not None:
        # TODO(ryand): This is a limitation of the UNet2DConditionModel API, not a fundamental incompatibility
        # between ControlNets and T2I-Adapters. We will try to fix this upstream in diffusers.
        #    raise Exception("ControlNet(s) and T2I-Adapter(s) cannot be used simultaneously (yet).")
        # elif control_data is not None:
        if control_data is not None:
            down_block_additional_residuals, mid_block_additional_residual = self.invokeai_diffuser.do_controlnet_step(
                control_data=control_data,
                sample=latent_model_input,
                timestep=timestep,
                step_index=step_index,
                total_step_count=total_step_count,
                conditioning_data=conditioning_data,
            )
        # elif t2i_adapter_data is not None:
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


        ################################ CUSTOM SUBSTEP  ################################
        noise_pred = self.custom_substep(
            sample=latent_model_input,
            t=t,  # TODO: debug how handled batched and non batched timesteps
            step_index=step_index,
            total_step_count=total_step_count,
            conditioning_data=conditioning_data,
            custom_module_data=self.custom_module_data,
            # extra:
            down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
            mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
        )
        ################################################################################

        # compute the previous noisy sample x_t -> x_t-1
        step_output = self.scheduler.step(noise_pred, timestep, latents, **conditioning_data.scheduler_args)

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