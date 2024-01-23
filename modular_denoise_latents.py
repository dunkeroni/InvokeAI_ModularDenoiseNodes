from invokeai.app.invocations.latent import DenoiseLatentsInvocation
from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline

from invokeai.app.invocations.baseinvocation import (
    BaseInvocationOutput,
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)

import math
from typing import Callable, List, Optional, Any, Union

import torch
from invokeai.backend.ip_adapter.unet_patcher import UNetPatcher
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData

from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    IPAdapterData,
    T2IAdapterData
)
from pydantic import BaseModel, Field

from .modular_decorators import get_noise_prediction_module, get_post_noise_guidance_module, get_pre_noise_guidance_module

import numpy as np

import inspect #TODO: get rid of this garbage

class ModuleData(BaseModel): #use child classes for different module types
    name: str = Field(description="Name of the module")
    module_type: str = Field(description="Type of the module")
    module: str = Field(description="Name of the module function")
    module_kwargs: dict | None = Field(description="Keyword arguments to pass to the module function")

class NP_ModuleData(ModuleData):
    module_type: str = Field(default="noise_pred", description="Type of the module")

class PreG_ModuleData(ModuleData):
    module_type: str = Field(default="pre_noise_guidance", description="Type of the module")

class PoG_ModuleData(ModuleData):
    module_type: str =  Field(default="post_noise_guidance", description="Type of the module")


@invocation_output("module_data_output")
class ModuleDataOutput(BaseInvocationOutput):
    module_data_output: ModuleData | None = OutputField(
        title="Module Data Output",
        description="Information for calling the module in denoise latents step()"
    )
@invocation_output("np_module_data_output")
class NP_ModuleDataOutput(ModuleDataOutput):
    module_data_output: ModuleData | None = OutputField(
        title="[NP] Module Data",
        description="Noise Prediction module data"
    )
@invocation_output("preg_module_data_output")
class PreG_ModuleDataOutput(ModuleDataOutput):
    module_data_output: ModuleData | None = OutputField(
        title="[PreG] Module Data",
        description="PRE-Noise Guidance module data"
    )
@invocation_output("pog_module_data_output")
class PoG_ModuleDataOutput(ModuleDataOutput):
    module_data_output: ModuleData | None = OutputField(
        title="[PoG] Module Data",
        description="POST-Noise Guidance module data"
    )

@invocation_output("module_data_collection_output")
class ModuleDataCollectionOutput(BaseInvocationOutput):
    module_data_output: list[ModuleData] | None = OutputField(
        title="Collected Module Data",
        description="Information for calling the modules in denoise latents"
    )

@invocation("module_collection",
            title="Module Collection",
            tags=["modular", "collection"],
            category="modular",
            version="1.0.0"
            )
class ModuleCollectionInvocation(BaseInvocation):
    """Collects multiple module types together"""
    pre_noise_module: ModuleData = InputField(
        default=None,
        description="Information for calling the module in denoise latents step()",
        title="[PreG] Module Data",
        input=Input.Connection,
    )
    noise_pred_module: ModuleData = InputField(
        default=None,
        description="Information for calling the module in denoise latents step()",
        title="[NP] Module Data",
        input=Input.Connection,
    )
    pog_noise_module: ModuleData = InputField(
        default=None,
        description="Information for calling the module in denoise latents step()",
        title="[PoG] Module Data",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> ModuleDataCollectionOutput:
        #make sure list is always list[ModuleData] and does not included Nones
        modules_list = []
        if self.pre_noise_module is not None:
            modules_list.append(self.pre_noise_module)
        if self.noise_pred_module is not None:
            modules_list.append(self.noise_pred_module)
        if self.pog_noise_module is not None:
            modules_list.append(self.pre_noise_module)
        
        if modules_list == []:
            return ModuleDataCollectionOutput(
                module_data_output=None
            )

        return ModuleDataCollectionOutput(
            module_data_output=modules_list
        )


#INHERIT to add custom module input
@invocation(
    "modular_denoise_latents",
    title="Modular Denoise Latents",
    tags=["modular", "generate", "denoise", "latents"],
    category="modular",
    version="1.5.0",
)
class Modular_DenoiseLatentsInvocation(DenoiseLatentsInvocation):
    module: Optional[Union[ModuleData, list[ModuleData]]] = InputField(
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

            UPDATE: There is a PR that should replace the need for context references with a solid API for nodes (PR #5491)
                    Once it is merged into a release, this hack can be removed.

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
            denoise_node=self, #NEW for when modules need to load input data
        )



class Modular_StableDiffusionGeneratorPipeline(StableDiffusionGeneratorPipeline):

    def __init__(self, *args, **kwargs):
        self.custom_module_data = kwargs.pop("custom_module_data", None)
        self.noise_pred_module_data: ModuleData = self.find_first_module_of_type(self.custom_module_data, "noise_pred")
        self.post_noise_guidance_module_data: ModuleData = self.find_first_module_of_type(self.custom_module_data, "post_noise_guidance")
        self.pre_noise_guidance_module_data: ModuleData = self.find_first_module_of_type(self.custom_module_data, "pre_noise_guidance")
        self.persistent_data = {} # for modules storing data between steps
        self.context: InvocationContext = kwargs.pop("context", None)
        self.denoise_node: Modular_DenoiseLatentsInvocation = kwargs.pop("denoise_node", None)
        super().__init__(*args, **kwargs)


    def find_first_module_of_type(self, module_data_input: Union[ModuleData, list[ModuleData]], module_type: str):
        """Find the first module of the given type in the pipeline."""
        if module_data_input is None:
            return None

        if isinstance(module_data_input, ModuleData):
            module_data_input = [module_data_input]
        
        for module_data in module_data_input:
            if module_data.module_type == module_type:
                return module_data
        return None # no module of given type found


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

        if self.pre_noise_guidance_module_data is not None:
            # invoke custom module
            pre_module_func: Callable = get_pre_noise_guidance_module(self.pre_noise_guidance_module_data.module)
            pre_module_kwargs = self.pre_noise_guidance_module_data.module_kwargs
            latents = pre_module_func(
                self=self,
                latents=latents,
                t=t,
                step_index=step_index,
                total_step_count=total_step_count,
                module_kwargs = pre_module_kwargs,
            )

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


        # default to standard diffusion pipeline result
        if self.noise_pred_module_data is None:
            module_func: Callable = get_noise_prediction_module("standard_unet_step_module")
            module_kwargs = {}
        else:
            # invoke custom module
            module_func: Callable = get_noise_prediction_module(self.noise_pred_module_data.module)
            module_kwargs = self.noise_pred_module_data.module_kwargs
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
        
        prev_sample = step_output["prev_sample"]

        if self.post_noise_guidance_module_data is not None:
            # invoke custom module
            post_module_func: Callable = get_post_noise_guidance_module(self.post_noise_guidance_module_data.module)
            post_module_kwargs = self.post_noise_guidance_module_data.module_kwargs
            modified_step_output = post_module_func(
                self=self,
                step_output=prev_sample,
                t=t,
                step_index=step_index,
                total_step_count=total_step_count,
                module_kwargs = post_module_kwargs,
            )
            
            step_output["prev_sample"] = modified_step_output

        # restore internal counter
        if self.scheduler.order == 2:
            self.scheduler._index_counter[timestep.item()] += 1

        return step_output


def are_like_tensors(a: torch.Tensor, b: object) -> bool:
    return isinstance(b, torch.Tensor) and (a.size() == b.size())