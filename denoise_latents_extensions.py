from abc import ABC, abstractmethod
import torch
from contextlib import ExitStack
from typing import Union, Type, Any, Callable
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from dataclasses import dataclass, field
from invokeai.app.invocations.t2i_adapter import T2IAdapterField
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.invocation_api import (
    InvocationContext,
    ConditioningField,
    LatentsField,
    UNetField,
    OutputField,
    invocation_output,
    BaseInvocationOutput,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    IPAdapterData,
    TextConditioningData,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    StableDiffusionGeneratorPipeline,
    T2IAdapterData,
)
from pydantic import BaseModel, Field
from invokeai.backend.util.logging import info, warning, error

SD12X_EXTENSIONS = {}

def guidance_extension_12X(name: str):
    """Register a guidance extension class object under a string reference"""
    def decorator(cls: Type[DenoiseExtensionSD12X]):
        if name in SD12X_EXTENSIONS:
            raise ValueError(f"Extension {name} already registered")
        info(f"Registered extension {cls.__name__} as {name}")
        SD12X_EXTENSIONS[name] = cls
        return cls
    return decorator

@dataclass
class DenoiseLatentsInputs:
    positive_conditioning: Union[ConditioningField, list[ConditioningField]]
    negative_conditioning: Union[ConditioningField, list[ConditioningField]]
    noise: LatentsField | None
    latents: LatentsField | None
    steps: int
    cfg_scale: Union[float, list[float]]
    denoising_start: float
    denoising_end: float
    scheduler: str
    unet: UNetField
    control: Union[ControlField, list[ControlField]] | None
    ip_adapter: Union[IPAdapterField, list[IPAdapterField]] | None
    t2i_adapter: Union[T2IAdapterField, list[T2IAdapterField]] | None

    def copy(self):
        return DenoiseLatentsInputs(
            positive_conditioning=self.positive_conditioning,
            negative_conditioning=self.negative_conditioning,
            noise=self.noise,
            latents=self.latents,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            denoising_start=self.denoising_start,
            denoising_end=self.denoising_end,
            scheduler=self.scheduler,
            unet=self.unet,
            control=self.control,
            ip_adapter=self.ip_adapter,
            t2i_adapter=self.t2i_adapter
        )

@dataclass
class DenoiseLatentsData:
    conditioning_data: TextConditioningData = None
    noise: torch.Tensor = None
    latents: torch.Tensor = None
    scaled_model_inputs: torch.Tensor = None
    timesteps: list[int] = field(default_factory=list)
    init_timestep: int = 0
    step_index: int = 0
    num_inference_steps: int = 0
    scheduler_step_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler: SchedulerMixin = None
    pipeline: StableDiffusionGeneratorPipeline = None
    unet: UNet2DConditionModel = None
    controlnet_data: (list[ControlNetData] | None) = None
    ip_adapter_data: (list[IPAdapterData] | None) = None
    t2i_adapter_data: (list[T2IAdapterData] | None) = None
    seed: int = 0
    misc: dict = field(default_factory=dict) # add custom data here for extensions that need to share

    def copy(self):
        return DenoiseLatentsData(
            conditioning_data=self.conditioning_data,
            noise=self.noise.clone() if self.noise is not None else None,
            latents=self.latents.clone() if self.latents is not None else None,
            scaled_model_inputs=self.scaled_model_inputs.clone() if self.scaled_model_inputs is not None else None,
            timesteps=self.timesteps.copy(),
            init_timestep=self.init_timestep,
            step_index=self.step_index,
            num_inference_steps=self.num_inference_steps,
            scheduler_step_kwargs=self.scheduler_step_kwargs.copy(),
            scheduler=self.scheduler,
            pipeline=self.pipeline,
            unet=self.unet,
            controlnet_data=self.controlnet_data,
            ip_adapter_data=self.ip_adapter_data,
            t2i_adapter_data=self.t2i_adapter_data,
            seed=self.seed,
            misc=self.misc.copy()
        )

class GuidanceField(BaseModel):
    """Guidance information for extensions in the denoising process."""
    guidance_name: str = Field(description="The name of the guidance extension class")
    priority: int = Field(default=100, description="Execution order for multiple guidance. Lower numbers go first.")
    extension_kwargs: dict[str, Any] = Field(default={}, description="Keyword arguments for the guidance extension")

@invocation_output("guidance_module_output")
class GuidanceDataOutput(BaseInvocationOutput):
    guidance_data_output: GuidanceField | None = OutputField(
        title="Guidance Module",
        description="Information to alter the denoising process"
    )

class ExtensionHandlerSD12X:
    """
    Extension Handler for the DenoiseLatentsSD12X class
    Starts with a list of guidance fields, instances and calls them when needed
    """
    def __init__(self, context: InvocationContext, extensions: Union[list[GuidanceField], None], denoise_inputs: DenoiseLatentsInputs):
        self.extensions: list[DenoiseExtensionSD12X] = []
        modifies: dict[str, list[DenoiseExtensionSD12X]] = {}
        swaps: dict[str, DenoiseExtensionSD12X] = {}

        if extensions is None:
            extensions = [] # empty list if no extensions are provided
        elif not isinstance(extensions, list):
            extensions = [extensions]
        
        for extension in extensions:
            extension_class: Type[DenoiseExtensionSD12X] = SD12X_EXTENSIONS.get(extension.guidance_name)
            if extension_class:
                self.extensions.append(extension_class(denoise_inputs.copy(), extension.guidance_name, extension.priority, context, extension.extension_kwargs))
            else:
                raise ValueError(f"Extension {extension.guidance_name} not found in the registry")

        for extension in self.extensions:
            #add all the modify methods to the modifies dict keys
            for method in extension.list_modifies():
                if method not in modifies:
                    modifies[method] = []
            #add all the swap methods to the swaps dict keys
            for method in extension.list_swaps():
                if method not in swaps:
                    swaps[method] = extension
                else:
                    raise ValueError(f"Swap method {method} already defined by a competing extension. There can be only one!")
        
        # add a reference to each applicable extension in every method list
        for method in modifies:
            for extension in self.extensions:
                functiondict = extension.list_modifies()
                if method in functiondict:
                    modifies[method].append(extension)
            # sort the list by priority
            modifies[method].sort(key=lambda x: x.priority)
    
        self.modifies = modifies
        self.swaps = swaps
    
    def call_patches(self, unet_model: UNet2DConditionModel):
        """Call all the patch_model methods in order of priority"""
        for extension in self.extensions:
            extension.patch_model(unet_model=unet_model)
        return self # required for the with block
    
    def __enter__(self):
        for extension in self.extensions:
            extension.__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        for extension in self.extensions:
            extension.__exit__(exc_type, exc_value, traceback)
    
    def call_modifiers(self, method: str, **kwargs) -> DenoiseLatentsData:
        """Call all the modify methods in order of priority"""
        if method not in self.modifies:
            return # none of the current extensions modify at this point
        
        for extension in self.modifies[method]: #already sorted by priority
            modifier: Callable = extension.list_modifies()[method]
            if callable(modifier):
                modifier(**kwargs) # usually kwargs is data, sometimes other arguments
            else:
                raise ValueError(f"Method {method} does not relate to a callable in extension {extension.extension_type} list_modifies()")
    
    def call_swap(self, method: str, default: Callable, **kwargs) -> DenoiseLatentsData:
        """Call the swap method if it exists, otherwise return the default function"""
        if method in self.swaps:
            swap: Callable = self.swaps[method].list_swaps()[method]
            if callable(swap):
                return swap(default=default, **kwargs)
            else:
                raise ValueError(f"Method {method} does not relate to a callable in extension {self.swaps[method].extension_type} list_swaps()")
        else:
            return default(**kwargs)
    
    def enter_contexts(self, exit_stack: ExitStack):
        """Enter the context of each extension"""
        for extension in self.extensions:
            exit_stack.enter_context(extension)
            extension.exit_stack = exit_stack


class DenoiseExtensionSD12X(ABC):

    def __init__(self, input_data: DenoiseLatentsInputs, extension_type: str, priority: int, context: InvocationContext, extension_kwargs: dict):
        """
        Do not modify: Use __post_init__ to handle extension-specific parameters
        During injection calls, extensions will be called in order of self.priority (ascending)
        self.input_data exists in case you need to access the original data from the calling node
        """
        self.extension_type = extension_type
        self.input_data = input_data
        self.priority = priority
        self.context = context
        self.exit_stack = None #Gets added before modify_data_before_denoising() is called
        self.__post_init__(**extension_kwargs)

    def __post_init__(self):
        """
        Called after the object is created.
        Override this method to perform additional initialization steps.
        """
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Override to clean up potentially loaded resources on exit.
        Not required for patch_model: that is already handled by the finally
        """

    def __enter__(self):
        """
        Override to set up special resources needed for the extension.
        """
        return None
    
    @abstractmethod
    def list_modifies(self) -> dict[str, Callable]:
        """
        A dict of all the modify methods that this extension provides.
        e.g. {'modify_latents_before_scaling': self.modify_latents_before_scaling,...}
        It is recommended that the injection names match the method names in this class.
        The list is accessed dynamically if you want to change the called function mid-process.
        """
        return {}
    
    @abstractmethod
    def list_swaps(self) -> dict[str, Callable]:
        """
        A dict of all the swap methods that this extension provides.
        e.g. ['swap_do_unet_step': self.swap_do_unet_step,...]
        It is recommended that the injection names match the method names in this class.
        This should remain static since the handler might break if the structure changes too much.
        """
        return {}
    
    def patch_model(self, unet_model: UNet2DConditionModel):
        """
        Do not modify: Place code in the modify_unet_model method to modify the unet model.
        Modify the unet model before it is used in the denoising process.
        """
        applied_modifier = False
        try:
            self.modify_unet_model(unet_model)
            applied_modifier = True

            yield

        finally:
            if applied_modifier:
                self.restore_unet_model(unet_model)
        return 

    def modify_unet_model(self, unet_model: UNet2DConditionModel):
        """
        Modify the unet model before it is used in the denoising process.
        """
        pass

    def restore_unet_model(self, unet_model: UNet2DConditionModel):
        """
        Restore the unet model after it is used in the denoising process.
        REQUIRED IF MODIFY_UNET_MODEL IS USED. Otherwise the in-memory model remains modified!
        """
        pass
    
    def modify_data_before_denoising(self, data: DenoiseLatentsData):
        """
        Modify the latents before the denoising process begins.
        """
        pass

    def modify_data_before_scaling(self, data: DenoiseLatentsData, t: torch.Tensor):
        """
        Samplers apply a scalar multiplication to the latents before predicting noise.
        This method allows you to modify the latents before this scaling is applied each step.
        Useful if the modifications need to align with image or color in the normal latent space.
        """
        pass

    def modify_data_before_noise_prediction(self, data: DenoiseLatentsData, t: torch.Tensor):
        """
        Last chance to modify latents before noise is predicted.
        Additional channels for inpaint models are added here.
        """
        pass

    def modify_result_before_callback(self, step_output, data: DenoiseLatentsData, t: torch.Tensor):
        """
        step_output.prev_sample is the current latents that will be used in the next step.
        if step_output.pred_original_sample is provided/modified, it will be used in the image preview for the user.
        """
        pass

    def modify_data_after_denoising(self, data: DenoiseLatentsData):
        """
        Final result of the latents after all steps are complete.
        """
        pass

    def swap_combine_noise(
            self,
            default: Callable,
            unconditioned_next_x: torch.Tensor,
            conditioned_next_x: torch.Tensor,
            guidance_scale: float,
        ) -> torch.Tensor:
        """Combine unconditional and conditional noise predictions"""
        return default(unconditioned_next_x, conditioned_next_x, guidance_scale)
    
    def swap_do_unet_step(
            self,
            default: Callable,
            sample: torch.Tensor,
            timestep: torch.Tensor,  # TODO: debug how handled batched and non batched timesteps
            step_index: int,
            total_step_count: int,
            conditioning_data: TextConditioningData,
            ip_adapter_data: IPAdapterData,
            down_block_additional_residuals: list[torch.Tensor],  # for ControlNet
            mid_block_additional_residual: torch.Tensor,  # for ControlNet
            down_intrablock_additional_residuals: list[torch.Tensor],  # for T2I-Adapter
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a step in the UNet model, returns unconditional and conditional noise predictions"""
        return default(
            sample,
            timestep,
            step_index,
            total_step_count,
            conditioning_data,
            ip_adapter_data,
            down_block_additional_residuals,
            mid_block_additional_residual,
            down_intrablock_additional_residuals
        )
        