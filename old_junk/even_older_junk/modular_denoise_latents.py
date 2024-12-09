from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Union

import torch

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import IPAdapterData, TextConditioningData

from invokeai.app.invocations.latent import DenoiseLatentsInvocation
from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline

from invokeai.invocation_api import (
    Input,
    InputField,
    InvocationContext,
    invocation,
)

from .models import ModuleData


from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    T2IAdapterData
)

from .modular_decorators import get_noise_prediction_module, get_post_noise_guidance_module, get_pre_noise_guidance_module

import inspect #TODO: get rid of this garbage

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
    @torch.inference_mode()
    def step(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        conditioning_data: TextConditioningData,
        step_index: int,
        total_step_count: int,
        scheduler_step_kwargs: dict[str, Any],
        additional_guidance: List[Callable] = None,
        control_data: List[ControlNetData] = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
    ):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]
        if additional_guidance is None:
            additional_guidance = []

        # one day we will expand this extension point, but for now it just does denoise masking
        for guidance in additional_guidance:
            latents = guidance(latents, timestep)

        if self.pre_noise_guidance_module_data is not None:
            # invoke custom module
            pre_module_func: Callable = get_pre_noise_guidance_module(self.pre_noise_guidance_module_data.module)
            pre_module_kwargs = self.pre_noise_guidance_module_data.module_kwargs
            latents_type = latents.dtype
            latents = pre_module_func(
                self=self,
                latents=latents,
                t=t,
                step_index=step_index,
                total_step_count=total_step_count,
                module_kwargs = pre_module_kwargs,
            )
            latents = latents.to(dtype=latents_type)

        # TODO: should this scaling happen here or inside self._unet_forward?
        #     i.e. before or after passing it to InvokeAIDiffuserComponent
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)


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
            ip_adapter_data=ip_adapter_data,
            conditioning_data=conditioning_data,
            module_kwargs = module_kwargs,
            control_data=control_data, # passed down for tiling nodes to recalculate control blocks
            t2i_adapter_data=t2i_adapter_data, # passed down for tiling nodes to recalculate t2i blocks
        )

        # compute the previous noisy sample x_t -> x_t-1
        step_output = self.scheduler.step(noise_pred, timestep, latents, **scheduler_step_kwargs)

        # TODO: discuss injection point options. For now this is a patch to get progress images working with inpainting again.
        for guidance in additional_guidance:
            # apply the mask to any "denoised" or "pred_original_sample" fields
            if hasattr(step_output, "denoised"):
                step_output.pred_original_sample = guidance(step_output.denoised, self.scheduler.timesteps[-1])
            elif hasattr(step_output, "pred_original_sample"):
                step_output.pred_original_sample = guidance(
                    step_output.pred_original_sample, self.scheduler.timesteps[-1]
                )
            else:
                step_output.pred_original_sample = guidance(latents, self.scheduler.timesteps[-1])

        prev_sample: torch.Tensor = step_output["prev_sample"] #TODO: Check that this works for all samplers
        latent_type = prev_sample.dtype
        if self.post_noise_guidance_module_data is not None:
            # invoke custom module
            post_module_func: Callable = get_post_noise_guidance_module(self.post_noise_guidance_module_data.module)
            post_module_kwargs = self.post_noise_guidance_module_data.module_kwargs
            modified_step_output: torch.Tensor = post_module_func(
                self=self,
                step_output=prev_sample,
                t=t,
                step_index=step_index,
                total_step_count=total_step_count,
                module_kwargs = post_module_kwargs,
            )
            
            step_output["prev_sample"] = modified_step_output.to(dtype=latent_type)
        
        return step_output


def are_like_tensors(a: torch.Tensor, b: object) -> bool:
    return isinstance(b, torch.Tensor) and (a.size() == b.size())