from .modular_decorators import module_noise_pred, get_noise_prediction_module
from .modular_denoise_latents import Modular_StableDiffusionGeneratorPipeline, ModuleData, NP_ModuleDataOutput, NP_ModuleData

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData
from invokeai.backend.stable_diffusion.diffusers_pipeline import ControlNetData, T2IAdapterData
from invokeai.backend.stable_diffusion.diffusion.shared_invokeai_diffusion import PostprocessingSettings
from invokeai.app.invocations.primitives import LatentsField, ImageField
from invokeai.app.invocations.compel import ConditioningField
from pydantic import BaseModel, Field
import torch
from typing import Literal, Optional, Callable, List, Union
from PIL import ImageFilter
import random
import numpy as np
import torch.nn.functional as F

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Input,
    InputField,
    OutputField,
    InvocationContext,
    UIType,
    invocation,
    invocation_output,
)


def resolve_module(module_dict: dict | None) -> tuple[Callable, dict]:
    """Resolve a module from a module dict. Handles None case automatically. """
    if module_dict is None:
        return get_noise_prediction_module(None), {}
    else:
        return get_noise_prediction_module(module_dict["module"]), module_dict["module_kwargs"]


####################################################################################################
# Standard UNet Step Module
####################################################################################################
"""
Fallback module for the noise prediction pipeline. This module is used when no other module is specified.
"""
@module_noise_pred("standard_unet_step_module")
def standard_do_unet_step(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor, #result of previous step
    t: torch.Tensor,
    conditioning_data: ConditioningData,
    step_index: int,
    total_step_count: int,
    control_data: List[ControlNetData] = None,
    t2i_adapter_data: list[T2IAdapterData] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = t[0]
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)
        
        # Handle ControlNet(s)
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        down_intrablock_additional_residuals = None
        if control_data is not None:
            down_block_additional_residuals, mid_block_additional_residual = self.invokeai_diffuser.do_controlnet_step(
                control_data=control_data,
                sample=latent_model_input,
                timestep=timestep,
                step_index=step_index,
                total_step_count=total_step_count,
                conditioning_data=conditioning_data,
            )
        
        # and T2I-Adapter(s)
        down_intrablock_additional_residuals = self.get_t2i_intrablock(t2i_adapter_data, step_index, total_step_count)

        # result from calling object's default pipeline
        # extra kwargs get dropped here, so pass whatever you like down the chain
        uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
            sample=latent_model_input,
            timestep=t,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
            mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
        )

        guidance_scale = conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[step_index]

        noise_pred = self.invokeai_diffuser._combine(uc_noise_pred, c_noise_pred, guidance_scale)
        guidance_rescale_multiplier = conditioning_data.guidance_rescale_multiplier
        if guidance_rescale_multiplier > 0:
            noise_pred = self._rescale_cfg(
                noise_pred,
                c_noise_pred,
                guidance_rescale_multiplier,
            )

        return noise_pred, latents

@invocation("standard_unet_step_module",
    title="Standard UNet Step Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class StandardStepModuleInvocation(BaseInvocation):
    """NP_MOD: InvokeAI standard noise prediction."""
    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Standard UNet Step Module",
            module="standard_unet_step_module",
            module_kwargs={},
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )


####################################################################################################
# Perp Negative
# From: https://perp-neg.github.io/
####################################################################################################
def get_perpendicular_component(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape
    return x - ((torch.mul(x, y).sum())/(torch.norm(y)**2)) * y

@module_noise_pred("perp_neg_unet_step")
def perp_neg_do_unet_step(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor, #result of previous step
    t: torch.Tensor,
    conditioning_data: ConditioningData,
    step_index: int,
    total_step_count: int,
    module_kwargs: dict | None,
    control_data: List[ControlNetData] = None,
    t2i_adapter_data: list[T2IAdapterData] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = t[0]
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)
        module_id = module_kwargs["module_id"]

        # check for saved unconditional
        unconditional_conditioning_data = self.check_persistent_data(module_id, "unconditional_conditioning_data")
        if unconditional_conditioning_data is None:
            # format conditioning data
            unconditional_name: str = module_kwargs["unconditional_name"]
            unconditional_data = self.context.services.latents.get(unconditional_name)
            c = unconditional_data.conditionings[0].to(device=latents.device, dtype=latents.dtype)
            extra_conditioning_info = c.extra_conditioning

            unconditional_conditioning_data = ConditioningData(
                unconditioned_embeddings=c,
                text_embeddings=conditioning_data.text_embeddings,
                guidance_scale=conditioning_data.guidance_scale,
                guidance_rescale_multiplier=conditioning_data.guidance_rescale_multiplier,
                extra=extra_conditioning_info,
                postprocessing_settings=PostprocessingSettings(
                    threshold=0.0,  # threshold,
                    warmup=0.2,  # warmup,
                    h_symmetry_time_pct=None,  # h_symmetry_time_pct,
                    v_symmetry_time_pct=None,  # v_symmetry_time_pct,
                ),
            )
            self.set_persistent_data(module_id, "unconditional_conditioning_data", unconditional_conditioning_data)

        
        # Handle ControlNet(s)
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        down_intrablock_additional_residuals = None
        if control_data is not None:
            down_block_additional_residuals, mid_block_additional_residual = self.invokeai_diffuser.do_controlnet_step(
                control_data=control_data,
                sample=latent_model_input,
                timestep=timestep,
                step_index=step_index,
                total_step_count=total_step_count,
                conditioning_data=conditioning_data,
            )
        
        # and T2I-Adapter(s)
        down_intrablock_additional_residuals = self.get_t2i_intrablock(t2i_adapter_data, step_index, total_step_count)

        # result from calling object's default pipeline
        # extra kwargs get dropped here, so pass whatever you like down the chain
        negative_noise_pred, positive_noise_pred = self.invokeai_diffuser.do_unet_step(
            sample=latent_model_input,
            timestep=t,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
            mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
        )

        # result from calling object's default pipeline with module's unconditional conditioning
        # NOTE: it is possible to get noise prediction from just the unconditional, and only take 50% longer instead of 100%
        # BUT: that's burried a few layers deeper than I want to customize right now.
        unconditional_noise_pred, _ = self.invokeai_diffuser.do_unet_step(
            sample=latent_model_input,
            timestep=t,
            conditioning_data=unconditional_conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
            mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
        )

        main = positive_noise_pred - unconditional_noise_pred
        e_i = negative_noise_pred - unconditional_noise_pred
        perp = -1 * get_perpendicular_component(e_i, main)

        guidance_scale = conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[step_index]

        #noise_pred = self.invokeai_diffuser._combine(uc_noise_pred, perp_noise_pred, guidance_scale)
        noise_pred = unconditional_noise_pred + guidance_scale * (main + perp)
        guidance_rescale_multiplier = conditioning_data.guidance_rescale_multiplier
        if guidance_rescale_multiplier > 0:
            noise_pred = self._rescale_cfg(
                noise_pred,
                positive_noise_pred,
                guidance_rescale_multiplier,
            )

        return noise_pred, latents

@invocation("perp_neg_unet_step_module",
    title="Perp Negative",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class PerpNegStepModuleInvocation(BaseInvocation):
    """NP_MOD: Perp Negative noise prediction."""
    unconditional: ConditioningField = InputField(
        description="EMPTY CONDITIONING GOES HERE",
        input=Input.Connection, 
    )
    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Perp Negative",
            module="perp_neg_unet_step",
            module_kwargs={
                "unconditional_name": self.unconditional.conditioning_name,
                "module_id": self.id,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )


####################################################################################################
# MultiDiffusion Sampling
# From: https://multidiffusion.github.io/
####################################################################################################
def get_views(height, width, window_size=128, stride=64, random_jitter=False):
    # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
    # if panorama's height/width < window_size, num_blocks of height/width should return 1
    num_blocks_height = int((height - window_size) / stride - 1e-6) + 2 if height > window_size else 1
    num_blocks_width = int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size

        if h_end > height:
            h_start = int(h_start + height - h_end)
            h_end = int(height)
        if w_end > width:
            w_start = int(w_start + width - w_end)
            w_end = int(width)
        if h_start < 0:
            h_end = int(h_end - h_start)
            h_start = 0
        if w_start < 0:
            w_end = int(w_end - w_start)
            w_start = 0

        if random_jitter:
            jitter_range = (window_size - stride) // 4
            w_jitter = 0
            h_jitter = 0
            if (w_start != 0) and (w_end != width):
                w_jitter = random.randint(-jitter_range, jitter_range)
            elif (w_start == 0) and (w_end != width):
                w_jitter = random.randint(-jitter_range, 0)
            elif (w_start != 0) and (w_end == width):
                w_jitter = random.randint(0, jitter_range)
            if (h_start != 0) and (h_end != height):
                h_jitter = random.randint(-jitter_range, jitter_range)
            elif (h_start == 0) and (h_end != height):
                h_jitter = random.randint(-jitter_range, 0)
            elif (h_start != 0) and (h_end == height):
                h_jitter = random.randint(0, jitter_range)
            h_start += (h_jitter + jitter_range)
            h_end += (h_jitter + jitter_range)
            w_start += (w_jitter + jitter_range)
            w_end += (w_jitter + jitter_range)
        
        views.append((int(h_start), int(h_end), int(w_start), int(w_end)))
    return views

def crop_residuals(residual: List | torch.Tensor | None, view: tuple[int, int, int, int]):
    if residual is None:
        print("residual is None")
        return None
    if isinstance(residual, list):
        print(f"list of residuals: {len(residual)}")
        return [crop_residuals(r, view) for r in residual]
    else:
        h_start, h_end, w_start, w_end = view
        print(f"new residual shape: {residual[:, :, h_start:h_end, w_start:w_end].shape}")
        return residual[:, :, h_start:h_end, w_start:w_end]

@module_noise_pred("multidiffusion_sampling")
def multidiffusion_sampling(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor,
    module_kwargs: dict | None,
    control_data: List[ControlNetData] = None,
    t2i_adapter_data: list[T2IAdapterData] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    height = latents.shape[2]
    width = latents.shape[3]
    window_size = module_kwargs["tile_size"] // 8
    stride = module_kwargs["stride"] // 8
    pad_mode = module_kwargs["pad_mode"]
    enable_jitter = module_kwargs["enable_jitter"]
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])

    views = get_views(height, width, stride=stride, window_size=window_size, random_jitter=enable_jitter)
    if enable_jitter:
        jitter_range = (window_size - stride) // 4
        latents_pad = F.pad(latents, (jitter_range, jitter_range, jitter_range, jitter_range), pad_mode, 0)

    else:
        jitter_range = 0
        latents_pad = latents

    count_local = torch.zeros_like(latents_pad)
    value_local = torch.zeros_like(latents_pad)
    count_local_latents = torch.zeros_like(latents_pad)
    value_local_latents = torch.zeros_like(latents_pad)
   
    for j, view in enumerate(views):
        h_start, h_end, w_start, w_end = view
        latents_for_view = latents_pad[:, :, h_start:h_end, w_start:w_end]

        _control_data = None
        # crop control data list into tiles
        if control_data is not None:
            _control_data = []
            for c in control_data:
                if enable_jitter:
                    _image_tensor = F.pad(c.image_tensor, (jitter_range*8, jitter_range*8, jitter_range*8, jitter_range*8), pad_mode, 0)
                else:
                    _image_tensor = c.image_tensor
                _control_data.append(ControlNetData(
                    model=c.model,
                    image_tensor=_image_tensor[:, :, h_start*8:h_end*8, w_start*8:w_end*8], #control tensor is in image space
                    weight=c.weight,
                    begin_step_percent=c.begin_step_percent,
                    end_step_percent=c.end_step_percent,
                    control_mode=c.control_mode,
                    resize_mode=c.resize_mode,
                ))
        
        # crop t2i adapter data list into tiles
        _t2i_adapter_data: list[T2IAdapterData] = []
        if t2i_adapter_data is not None:

            for a in t2i_adapter_data:
                tensorlist = a.adapter_state #for some reason this is a List and not a dict, despite what the class definition says!
                _tensorlist = []
                for tensor in tensorlist:
                    scale = height // tensor.shape[2] # SDXL and 1.5 handle differently, one will be 1/2, 1/2, 1/4, 1/4, the other is 1/1, 1/2, 1/4, 1/8
                    if enable_jitter:
                        #hopefully 8 is a common factor of your jitter range and your latent size...
                        _tensor = F.pad(tensor, (jitter_range//scale, jitter_range//scale, jitter_range//scale, jitter_range//scale), pad_mode, 0)
                    else:
                        _tensor = tensor
                    _tensorlist.append(_tensor[:, :, h_start//scale:h_end//scale, w_start//scale:w_end//scale])
                _t2i_adapter_data.append(T2IAdapterData(
                    adapter_state=_tensorlist,
                    weight=a.weight,
                    begin_step_percent=a.begin_step_percent,
                    end_step_percent=a.end_step_percent,
                ))

        noise_pred, original_view_latents = sub_module(
            self=self,
            latents=latents_for_view,
            module_kwargs=sub_module_kwargs,
            control_data=_control_data,
            t2i_adapter_data=_t2i_adapter_data,
            **kwargs,
        )

        # get prediction from output
        value_local[:, :, h_start:h_end, w_start:w_end] += noise_pred #step_output.prev_sample.detach().clone()
        count_local[:, :, h_start:h_end, w_start:w_end] += 1
        value_local_latents[:, :, h_start:h_end, w_start:w_end] += original_view_latents
        count_local_latents[:, :, h_start:h_end, w_start:w_end] += 1

    value_local_crop = value_local[: ,:, jitter_range: jitter_range + height, jitter_range: jitter_range + width]
    count_local_crop = count_local[: ,:, jitter_range: jitter_range + height, jitter_range: jitter_range + width]
    value_local_latents_crop = value_local_latents[: ,:, jitter_range: jitter_range + height, jitter_range: jitter_range + width]
    count_local_latents_crop = count_local_latents[: ,:, jitter_range: jitter_range + height, jitter_range: jitter_range + width]

    combined_noise_pred = (value_local_crop / count_local_crop)
    combined_latents = (value_local_latents_crop / count_local_latents_crop)

    return combined_noise_pred, combined_latents


MD_PAD_MODES = Literal[
    "constant",
    "reflect",
    "replicate",
]

@invocation("multidiffusion_sampling_module",
    title="MultiDiffusion Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class MultiDiffusionSamplingModuleInvocation(BaseInvocation):
    """NP_MOD: MultiDiffusion tiled sampling. NOT compatible with t2i adapters."""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="[NP] SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    tile_size: int = InputField(
        title="Tile Size",
        description="Size of the tiles during noise prediction",
        ge=128,
        default=512,
        multiple_of=8,
    )
    stride: int = InputField(
        title="Stride",
        description="The spacing between the starts of tiles during noise prediction (recommend=tile_size/2)",
        ge=64,
        default=256,
        multiple_of=64,
    )
    pad_mode: MD_PAD_MODES = InputField(
        title="Padding Mode",
        description="Padding mode for extending the borders of the latent",
        default="reflect",
        input=Input.Direct,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="MultiDiffusion Sampling Step module",
            module="multidiffusion_sampling",
            module_kwargs={
                "sub_module": self.sub_module,
                "tile_size": self.tile_size,
                "stride": self.stride,
                "pad_mode": self.pad_mode,
                "enable_jitter": True,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Dilated Sampling
# From: https://ruoyidu.github.io/demofusion/demofusion.html
####################################################################################################
def gaussian_kernel(kernel_size=3, sigma=1.0, channels=3):
    x_coord = torch.arange(kernel_size)
    gaussian_1d = torch.exp(-(x_coord - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    kernel = gaussian_2d[None, None, :, :].repeat(channels, 1, 1, 1)
    
    return kernel

def gaussian_filter(latents, kernel_size=3, sigma=1.0):
    channels = latents.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(latents.device, latents.dtype)
    blurred_latents = F.conv2d(latents, kernel, padding=kernel_size//2, groups=channels)
    
    return blurred_latents

@module_noise_pred("dilated_sampling")
def dilated_sampling(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor,
    t: torch.Tensor,
    module_kwargs: dict | None,
    control_data: List[ControlNetData] = None, #prevent from being passed in kwargs
    t2i_adapter_data: list[T2IAdapterData] = None, #prevent from being passed in kwargs
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    gaussian_decay_rate = module_kwargs["gaussian_decay_rate"]
    dilation_scale = module_kwargs["dilation_scale"]
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (self.scheduler.config.num_train_timesteps - t) / self.scheduler.config.num_train_timesteps)).cpu()
    sigma = cosine_factor ** gaussian_decay_rate + 1e-2

    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])
        
    total_noise_pred = torch.zeros_like(latents)
    total_original_latents = torch.zeros_like(latents)
    std_, mean_ = latents.std(), latents.mean()
    blurred_latents = gaussian_filter(latents, kernel_size=(2*dilation_scale-1), sigma=sigma)
    blurred_latents = (blurred_latents - blurred_latents.mean()) / blurred_latents.std() * std_ + mean_
    for h in range(dilation_scale):
        for w in range(dilation_scale):
            #get interlaced subsample
            subsample = blurred_latents[:, :, h::dilation_scale, w::dilation_scale]
            noise_pred, original_latents = sub_module(
                self=self,
                latents=subsample,
                t=t,
                module_kwargs=sub_module_kwargs,
                control_data=None,
                t2i_adapter_data=None,
                **kwargs,
            )

            # insert subsample noise prediction into total tensor
            total_noise_pred[:, :, h::dilation_scale, w::dilation_scale] = noise_pred
            total_original_latents[:, :, h::dilation_scale, w::dilation_scale] = original_latents
    return total_noise_pred, total_original_latents

@invocation("dilated_sampling_module",
    title="Dilated Sampling Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class DilatedSamplingModuleInvocation(BaseInvocation):
    """NP_MOD: Dilated Sampling"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each interlaced noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    dilation_scale: int = InputField(
        title="Dilation Factor",
        description="The dilation scale to use when creating interlaced latents (e.g. '2' will split every 2x2 square among 4 latents)",
        ge=1,
        default=2,
    )
    gaussian_decay_rate: float = InputField(
        title="Gaussian Decay Rate",
        description="The decay rate to use when blurring the combined latents. Higher values will result in more blurring in later timesteps.",
        ge=0,
        default=1,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Dilated Sampling Step module",
            module="dilated_sampling",
            module_kwargs={
                "sub_module": self.sub_module,
                "dilation_scale": self.dilation_scale,
                "gaussian_decay_rate": self.gaussian_decay_rate,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Transfer Function: Cosine Decay
# From: https://ruoyidu.github.io/demofusion/demofusion.html
####################################################################################################
@module_noise_pred("cosine_decay_transfer")
def cosine_decay_transfer(
    self: Modular_StableDiffusionGeneratorPipeline,
    t: torch.Tensor,
    module_kwargs: dict | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    decay_rate = module_kwargs["decay_rate"]
    sub_module_1, sub_module_1_kwargs = resolve_module(module_kwargs["sub_module_1"])
    sub_module_2, sub_module_2_kwargs = resolve_module(module_kwargs["sub_module_2"])

    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (self.scheduler.config.num_train_timesteps - t) / self.scheduler.config.num_train_timesteps))
    c2 = 1 - cosine_factor ** decay_rate

    pred_1, latents_1 = sub_module_1(
        self=self,
        t=t,
        module_kwargs=sub_module_1_kwargs,
        **kwargs,
    )

    pred_2, latents_2 = sub_module_2(
        self=self,
        t=t,
        module_kwargs=sub_module_2_kwargs,
        **kwargs,
    )
    total_noise_pred = torch.lerp(pred_1, pred_2, c2.to(pred_1.device, dtype=pred_1.dtype))
    total_original_latents = torch.lerp(latents_1, latents_2, c2.to(latents_1.device, dtype=latents_1.dtype))

    return total_noise_pred, total_original_latents

@invocation("cosine_decay_transfer_module",
    title="Cosine Decay Transfer",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class CosineDecayTransferModuleInvocation(BaseInvocation):
    """NP_MOD: Smoothly changed modules based on remaining denoise"""
    sub_module_1: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the first noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModule 1",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    sub_module_2: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the second noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModule 2",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    decay_rate: float = InputField(
        title="Cosine Decay Rate",
        description="The decay rate to use when combining the two noise predictions. Higher values will shift the balance towards the second noise prediction sooner",
        ge=0,
        default=1,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Cosine Decay Transfer module",
            module="cosine_decay_transfer",
            module_kwargs={
                "sub_module_1": self.sub_module_1,
                "sub_module_2": self.sub_module_2,
                "decay_rate": self.decay_rate,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Transfer Function: Linear
####################################################################################################
@module_noise_pred("linear_transfer")
def linear_transfer(
    self: Modular_StableDiffusionGeneratorPipeline,
    t: torch.Tensor,
    step_index: int,
    module_kwargs: dict | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    start_step: int = module_kwargs["start_step"]
    end_step: int = module_kwargs["end_step"]
    sub_module_1, sub_module_1_kwargs = resolve_module(module_kwargs["sub_module_1"])
    sub_module_2, sub_module_2_kwargs = resolve_module(module_kwargs["sub_module_2"])
    
    linear_factor = (step_index - start_step) / (end_step - start_step)
    linear_factor = min(max(linear_factor, 0), 1)

    if linear_factor < 1:
        pred_1, latents_1 = sub_module_1(
            self=self,
            t=t,
            step_index=step_index,
            module_kwargs=sub_module_1_kwargs,
            **kwargs,
        )

    if linear_factor > 0:
        pred_2, latents_2 = sub_module_2(
            self=self,
            t=t,
            step_index=step_index,
            module_kwargs=sub_module_2_kwargs,
            **kwargs,
        )
    
    if linear_factor == 0:
        total_noise_pred = pred_1 # no need to lerp
        total_original_latents = latents_1
    elif linear_factor == 1:
        total_noise_pred = pred_2 # no need to lerp
        total_original_latents = latents_2
    else:
        total_noise_pred = torch.lerp(pred_1, pred_2, linear_factor)
        total_original_latents = torch.lerp(latents_1, latents_2, linear_factor)

    return total_noise_pred, total_original_latents

@invocation("linear_transfer_module",
    title="Linear Transfer",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class LinearTransferModuleInvocation(BaseInvocation):
    """NP_MOD: Smoothly change modules based on step."""
    sub_module_1: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the first noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModule 1",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    sub_module_2: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the second noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModule 2",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    start_step: int = InputField(
        title="Start Step",
        description="The step index at which to start using the second noise prediction",
        ge=0,
        default=0,
    )
    end_step: int = InputField(
        title="End Step",
        description="The step index at which to stop using the first noise prediction",
        ge=0,
        default=10,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Linear Transfer module",
            module="linear_transfer",
            module_kwargs={
                "sub_module_1": self.sub_module_1,
                "sub_module_2": self.sub_module_2,
                "start_step": self.start_step,
                "end_step": self.end_step,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )


####################################################################################################
# Transfer Function: Switch
####################################################################################################

@invocation("switch_transfer_module",
    title="Switch Transfer",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class SwitchTransferModuleInvocation(BaseInvocation):
    """NP_MOD: Switch between two pipelines at a specific step."""
    sub_module_1: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the first noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModule 1",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    sub_module_2: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the second noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModule 2",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    switch_step: int = InputField(
        title="Switch Step",
        description="The step index at which to switch from the first noise prediction to the second",
        ge=0,
        default=10,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Switch Transfer module",
            module="linear_transfer",
            module_kwargs={
                "sub_module_1": self.sub_module_1,
                "sub_module_2": self.sub_module_2,
                "start_step": self.switch_step - 1,
                "end_step": self.switch_step,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Transfer Function: Constant
####################################################################################################
@module_noise_pred("constant_transfer")
def constant_transfer(
    self: Modular_StableDiffusionGeneratorPipeline,
    module_kwargs: dict | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    sub_module_1, sub_module_1_kwargs = resolve_module(module_kwargs["sub_module_1"])
    sub_module_2, sub_module_2_kwargs = resolve_module(module_kwargs["sub_module_2"])
    ratio = module_kwargs["ratio"]

    pred_1, latents_1 = sub_module_1(
        self=self,
        module_kwargs=sub_module_1_kwargs,
        **kwargs,
    )

    pred_2, latents_2 = sub_module_2(
        self=self,
        module_kwargs=sub_module_2_kwargs,
        **kwargs,
    )

    total_noise_pred = torch.lerp(pred_1, pred_2, ratio)
    total_original_latents = torch.lerp(latents_1, latents_2, ratio)

    return total_noise_pred, total_original_latents

@invocation("constant_transfer_module",
    title="Constant Transfer",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class ConstantTransferModuleInvocation(BaseInvocation):
    """NP_MOD: Constantly use a ratio of two noise predictions."""
    sub_module_1: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the first noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModule 1",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    sub_module_2: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the second noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModule 2",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    ratio: float = InputField(
        title="Ratio",
        description="The ratio of the first noise prediction to the second. A ratio of 0.5 will use an even mix of both.",
        ge=0,
        le=1,
        default=0.5,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Constant Transfer module",
            module="constant_transfer",
            module_kwargs={
                "sub_module_1": self.sub_module_1,
                "sub_module_2": self.sub_module_2,
                "ratio": self.ratio,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Transfer Function: Parallel
####################################################################################################
"""takes in a list of submodules and runs them all in parallel, then averages the results"""
@module_noise_pred("parallel_transfer")
def parallel_transfer(
    self: Modular_StableDiffusionGeneratorPipeline,
    module_kwargs: dict | None,
    latents: torch.Tensor, #to make distinct clones
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    sub_modules: list[ModuleData] = module_kwargs["sub_modules"]

    noise_pred_list = []
    latent_list = []
    for i, sub in enumerate(sub_modules):
        sub_module, sub_module_kwargs = resolve_module(sub)
        pred, orig_sub_latent = sub_module(
            self=self,
            latents=latents.clone(), #clone to avoid modifying the original
            module_kwargs=sub_module_kwargs,
            **kwargs,
        )
        noise_pred_list.append(pred)
        latent_list.append(orig_sub_latent)
    total_noise_pred = torch.mean(torch.stack(noise_pred_list), dim=0)
    total_original_latents = torch.mean(torch.stack(latent_list), dim=0)

    return total_noise_pred, total_original_latents

@invocation("parallel_transfer_module",
    title="Parallel Transfer",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class ParallelTransferModuleInvocation(BaseInvocation):
    """NP_MOD: Run multiple noise predictions in parallel."""
    sub_modules: list[ModuleData] = InputField(
        default=[],
        description="The custom modules to use for each noise prediction. No connection will use the default pipeline.",
        title="[NP] SubModules",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Parallel Transfer module",
            module="parallel_transfer",
            module_kwargs={
                "sub_modules": self.sub_modules,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Tiled Denoise Latents
####################################################################################################
#Doesn't have it's own module function, relies on MultiDiffusion Sampling with jitter disabled.

@invocation("tiled_denoise_latents_module",
    title="Tiled Denoise Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class TiledDenoiseLatentsModuleInvocation(BaseInvocation):
    """NP_MOD: Denoise latents using tiled noise prediction"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="[NP] SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    tile_size: int = InputField(
        title="Tile Size",
        description="Size of the tiles during noise prediction",
        ge=128,
        default=512,
        multiple_of=8,
    )
    overlap: int = InputField(
        title="Overlap",
        description="The minimum amount of overlap between tiles during noise prediction",
        ge=0,
        default=64,
        multiple_of=8,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Tiled Denoise Latents module",
            module="multidiffusion_sampling",
            module_kwargs={
                "sub_module": self.sub_module,
                "tile_size": self.tile_size,
                "stride": self.tile_size - self.overlap,
                "enable_jitter": False,
                "pad_mode": None,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )


####################################################################################################
# Skip Residual 
# From: https://ruoyidu.github.io/demofusion/demofusion.html
####################################################################################################
"""Instead of denoising, synthetically noise an input latent to the noise level of the current timestep."""
@module_noise_pred("skip_residual")
def skip_residual(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor, #just to get the device
    t: torch.Tensor,
    module_kwargs: dict | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents_input: dict = module_kwargs["latent_input"] #gets serialized into a dict instead of a LatentsField for some reason
    noise_input: dict = module_kwargs["noise_input"] #gets serialized into a dict instead of a LatentsField for some reason
    module_id = module_kwargs["module_id"]

    #latents and noise are retrieved and stored in the pipeline object to avoid loading them from disk every step
    persistent_latent = self.check_persistent_data(module_id, "latent") #the latent from the original input on the module
    persistent_noise = self.check_persistent_data(module_id, "noise") #the noise from the original input on the module
    if persistent_latent is None: #load on first call
        persistent_latent = self.context.services.latents.get(latents_input["latents_name"]).to(latents.device)
        self.set_persistent_data(module_id, "latent", persistent_latent)
    if persistent_noise is None: #load on first call
        persistent_noise = self.context.services.latents.get(noise_input["latents_name"]).to(latents.device)
        self.set_persistent_data(module_id, "noise", persistent_noise)

    noised_latents = torch.lerp(persistent_latent, persistent_noise, ((t) / self.scheduler.config.num_train_timesteps).item())

    return torch.zeros_like(noised_latents), noised_latents

@invocation("skip_residual_module",
    title="Skip Residual Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class SkipResidualModuleInvocation(BaseInvocation):
    """NP_MOD: Skip Residual"""
    latent_input: LatentsField = InputField(
        title="Latent Input",
        description="The base latent to use for the noise prediction (usually the same as the input for img2img)",
        input=Input.Connection,
    )
    noise_input: LatentsField = InputField(
        title="Noise Input",
        description="The noise to add to the latent for the noise prediction (usually the same as the noise on the denoise latents node)",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Skip Residual module",
            module="skip_residual",
            module_kwargs={
                "latent_input": self.latent_input,
                "noise_input": self.noise_input,
                "module_id": self.id,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Sharpness
# From: https://github.com/lllyasviel/Fooocus/blob/176faf6f347b90866afe676fc9fb2c2d74587d7b/modules/patch.py
# GNU General Public License v3.0 https://github.com/lllyasviel/Fooocus/blob/176faf6f347b90866afe676fc9fb2c2d74587d7b/LICENSE
####################################################################################################
"""
Increase the sharpness of the image by applying a filter to the noise prediction and compute guidance influenced by the result.
"""
from .anisotropic import adaptive_anisotropic_filter
@module_noise_pred("fooocus_sharpness_module")
def fooocus_sharpness_unet_step(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor, #result of previous step
    t: torch.Tensor,
    conditioning_data: ConditioningData,
    step_index: int,
    total_step_count: int,
    module_kwargs: dict | None,
    control_data: List[ControlNetData] = None,
    t2i_adapter_data: list[T2IAdapterData] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = t[0]
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)
        sharpness = module_kwargs["sharpness"]
        
        # Handle ControlNet(s)
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        down_intrablock_additional_residuals = None
        if control_data is not None:
            down_block_additional_residuals, mid_block_additional_residual = self.invokeai_diffuser.do_controlnet_step(
                control_data=control_data,
                sample=latent_model_input,
                timestep=timestep,
                step_index=step_index,
                total_step_count=total_step_count,
                conditioning_data=conditioning_data,
            )
        
        # and T2I-Adapter(s)
        down_intrablock_additional_residuals = self.get_t2i_intrablock(t2i_adapter_data, step_index, total_step_count)

        # result from calling object's default pipeline
        # extra kwargs get dropped here, so pass whatever you like down the chain
        uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
            sample=latent_model_input,
            timestep=t,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
            mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
        )

        guidance_scale = conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[step_index]

        # The following code is adapted from lllyasveil's Fooocus project, linked above
        # It has been modified by @dunkeroni to work with the invokeai_diffuser pipeline components
        positive_eps = latent_model_input - c_noise_pred
        negative_eps = latent_model_input - uc_noise_pred

        alpha = 0.001 * sharpness * (1 - timestep.item()/999.0)
        positive_eps_degraded = adaptive_anisotropic_filter(x=positive_eps, g=c_noise_pred)
        positive_eps_degraded_weighted = torch.lerp(positive_eps, positive_eps_degraded, alpha)

        noise_pred = latent_model_input - self.invokeai_diffuser._combine(negative_eps, positive_eps_degraded_weighted, guidance_scale)
        # End of adapted code

        guidance_rescale_multiplier = conditioning_data.guidance_rescale_multiplier
        if guidance_rescale_multiplier > 0:
            noise_pred = self._rescale_cfg(
                noise_pred,
                c_noise_pred,
                guidance_rescale_multiplier,
            )

        return noise_pred, latents

@invocation("fooocus_sharpness_module",
    title="Sharpness Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class FooocusSharpnessModuleInvocation(BaseInvocation):
    """NP_MOD: Sharpness"""
    sharpness: float = InputField(
        title="Sharpness",
        description="The sharpness to apply to the noise prediction. Recommended Range: 2~30",
        gt=0,
        default=2,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        module = NP_ModuleData(
            name="Sharpness module",
            module="fooocus_sharpness_module",
            module_kwargs={
                "sharpness": self.sharpness,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Override Conditioning
####################################################################################################
"""
Replace the denoise latents conditioning data with a custom set of conditioning data.
"""

@module_noise_pred("override_conditioning")
def override_conditioning(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor,
    conditioning_data: ConditioningData,
    module_kwargs: dict | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])
    new_cfg = module_kwargs["cfg"]
    cfg_rescale = module_kwargs["cfg_rescale"]
    new_conditioning_data = self.check_persistent_data(module_kwargs["module_id"], "conditioning_data")
    if new_conditioning_data is None:
        positives: str = module_kwargs["positive_conditioning_data"]
        negatives: str = module_kwargs["negative_conditioning_data"]
        if positives is not None:
            c = self.context.services.latents.get(positives).conditionings[0].to(device=latents.device, dtype=latents.dtype)
            extra_conditioning_info = c.extra_conditioning
        else:
            c = conditioning_data.text_embeddings.conditionings[0].to(device=latents.device, dtype=latents.dtype)
            extra_conditioning_info = conditioning_data.extra
        
        if negatives is not None:
            uc = self.context.services.latents.get(negatives)
        else:
            uc = conditioning_data.unconditioned_embeddings
        
        new_conditioning_data = ConditioningData(
            unconditioned_embeddings=uc,
            text_embeddings=c,
            guidance_scale=new_cfg,
            guidance_rescale_multiplier=cfg_rescale,
            extra=extra_conditioning_info,
            postprocessing_settings=PostprocessingSettings(
                threshold=0.0,  # threshold,
                warmup=0.2,  # warmup,
                h_symmetry_time_pct=None,  # h_symmetry_time_pct,
                v_symmetry_time_pct=None,  # v_symmetry_time_pct,
            ),
        )
        self.set_persistent_data(module_kwargs["module_id"], "conditioning_data", new_conditioning_data)
    
    noise_pred, latents = sub_module(
        self=self,
        latents=latents,
        conditioning_data=new_conditioning_data,
        module_kwargs=sub_module_kwargs,
        **kwargs,
    )

    return noise_pred, latents

@invocation("override_conditioning_module",
    title="Override Conditioning Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class OverrideConditioningModuleInvocation(BaseInvocation):
    """NP_MOD: Override Conditioning"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="[NP] SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    positive_conditioning_data: Optional[ConditioningField] = InputField(
        default=None,
        description="The positive conditioning data to use for the noise prediction. No connection will use the default pipeline.",
        title="Positive Conditioning",
        input=Input.Connection,
    )
    negative_conditioning_data: Optional[ConditioningField] = InputField(
        default=None,
        description="The negative conditioning data to use for the noise prediction. No connection will use the default pipeline.",
        title="Negative Conditioning",
        input=Input.Connection,
    )
    cfg: float = InputField(
        title="CFG Scale",
        ge=1,
        default=7.5,
    )
    cfg_rescale: float = InputField(
        title="CFG Rescale Multiplier",
        ge=0,
        lt=1,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        c = self.positive_conditioning_data.conditioning_name if self.positive_conditioning_data is not None else None
        uc = self.negative_conditioning_data.conditioning_name if self.negative_conditioning_data is not None else None
        module = NP_ModuleData(
            name="Override Conditioning module",
            module="override_conditioning",
            module_kwargs={
                "sub_module": self.sub_module,
                "positive_conditioning_data": c,
                "negative_conditioning_data": uc,
                "cfg": self.cfg,
                "cfg_rescale": self.cfg_rescale,
                "module_id": self.id,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Regional Noise Prediction
####################################################################################################
class RegionalNoisePredictionData(BaseModel):
    region_mask_image: str = Field(description="Name of the saved image to use as a mask for the region")
    blur: float = Field(description="The amount to blur the mask")
    crop: bool = Field(description="Whether to crop the region to the size of the mask")
    crop_padding: int = Field(description="Padding (latent) to add to the region crop")
    weight: float = Field(description="Weight to apply to the region")
    sub_module: Optional[ModuleData] = Field(description="The custom module to use for the region. No connection will use the default pipeline.", default=None)

@invocation_output("regional_noise_pred_output")
class RegionalNoisePredictionOutput(BaseInvocationOutput):
    data: RegionalNoisePredictionData = OutputField(description="The regional noise prediction data")

@invocation("regional_module",
    title="Regional Guidance Info",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class RegionalNoiseDataModuleInvocation(BaseInvocation):
    """NP_MOD: Regional Info"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the region. No connection will use the default pipeline.",
        title="[NP] SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    region_mask: ImageField = InputField(
        title="Region Mask",
        description="The mask to use for the region",
    )
    blur: float = InputField(
        title="Blur",
        description="The amount to blur the mask",
        ge=0,
        default=0,
    )
    crop: bool = InputField(
        title="Crop",
        description="Whether to crop the region to the size of the mask",
        default=False,
    )
    crop_padding: int = InputField(
        title="Crop Padding",
        description="Padding (latent) to add to the region crop",
        default=0,
    )
    weight: float = InputField(
        title="Weight",
        description="Weight to apply to the region",
        default=1.0,
    )

    def invoke(self, context: InvocationContext) -> RegionalNoisePredictionOutput:
        module = RegionalNoisePredictionData(
            region_mask_image=self.region_mask.image_name,
            blur=self.blur,
            crop=self.crop,
            crop_padding=self.crop_padding,
            weight=self.weight,
            sub_module=self.sub_module,
        )

        return RegionalNoisePredictionOutput(
            data=module,
        )

@module_noise_pred("regional_noise_pred")
def regional_noise_pred(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor,
    t: torch.Tensor,
    module_kwargs: dict | None,
    control_data: List[ControlNetData] = None, #prevent from being passed in kwargs
    t2i_adapter_data: list[T2IAdapterData] = None, #prevent from being passed in kwargs
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Default Sub Module: Applied to the entire latents input
    region_info: list of RegionalNoisePredictionData
        Each region_info element has a submodule and a mask where it gets applied
    Results of the default and all regions are added together based on the region weights, then divided by the sum of the weights
    """
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["default_sub_module"])
    region_info: list[dict] = module_kwargs["region_info"] #gets serialized into a dict instead of a list
    default_weight = module_kwargs["default_weight"]

    default_noise_pred, default_latents = sub_module(
        self=self,
        latents=latents,
        t=t,
        module_kwargs=sub_module_kwargs,
        control_data=None,
        t2i_adapter_data=None,
        **kwargs,
    )
    total_noise_pred = default_noise_pred * default_weight
    total_latents = default_latents * default_weight
    total_weight = torch.ones_like(default_noise_pred) * default_weight

    #TODO: This whole persistent data setup is a bit of a mess. Something to consider during a rework towards object-based modules in the future.
    if self.check_persistent_data(module_kwargs["module_id"], "initialized") is None:
        # preprocess the masks
        masklist = []
        crop_tuples = []
        for region in region_info:
            mask = self.context.services.images.get_pil_image(region["region_mask_image"])
            if region["blur"] > 0:
                blur = ImageFilter.BoxBlur(region["blur"])
                mask = mask.filter(blur)
            mask = torch.from_numpy(np.asarray(mask.convert("L"))).to(latents.device)
            #scale the values between 0 and 1
            mask = mask / 255.0
            #repeat the mask so it has the same shape as the latents
            mask = mask.repeat(latents.shape[0], 4, 1, 1)
            #scale the mask to have the same X and Y dimensions as the latents
            mask = F.interpolate(mask, size=latents.shape[2:], mode="bilinear", align_corners=False)
            masklist.append(mask)
            self.set_persistent_data(module_kwargs["module_id"], "masklist", masklist)

            
            if region["crop"]:
                #determine rectangular bounds of the mask, any values < 1 are considered part of the mask
                mask_bounds = torch.nonzero(mask[0,0,:,:] < 1)
                #get the min and max values for each dimension
                min_x = torch.min(mask_bounds[:,0])
                max_x = torch.max(mask_bounds[:,0])
                min_y = torch.min(mask_bounds[:,1])
                max_y = torch.max(mask_bounds[:,1])
                """
                Fix boundaries:
                bounds should be shifted out by crop_padding
                max - min should be expanded to the nearest multiple of 8
                mins and maxs need to be adjusted to stay inside of the latents boundaries
                """
                min_x = max(min_x - region["crop_padding"], 0)
                max_x = min(max_x + region["crop_padding"], latents.shape[2])
                min_y = max(min_y - region["crop_padding"], 0)
                max_y = min(max_y + region["crop_padding"], latents.shape[3])
                #expand the to the nearest multiple of 8
                width = max_x - min_x
                width = width + (8 - width % 8)
                max_x = max(min_x + width, latents.shape[2])
                height = max_y - min_y
                height = height + (8 - height % 8)
                max_y = max(min_y + height, latents.shape[3])
                crop_tuples.append((min_x, max_x, min_y, max_y))
            else:
                crop_tuples.append((0, latents.shape[2], 0, latents.shape[3]))
        self.set_persistent_data(module_kwargs["module_id"], "crop_tuples", crop_tuples)

        self.set_persistent_data(module_kwargs["module_id"], "initialized", True)
    else:
        masklist = self.check_persistent_data(module_kwargs["module_id"], "masklist")
        crop_tuples = self.check_persistent_data(module_kwargs["module_id"], "crop_tuples")


    for i, region in enumerate(region_info):
        mask = masklist[i]
        sub_module, sub_module_kwargs = resolve_module(region["sub_module"])
        sub_latents = latents.clone() #clone to avoid modifying the original
        sub_crop = crop_tuples[i]
        sub_latents = sub_latents[:,:,sub_crop[0]:sub_crop[1],sub_crop[2]:sub_crop[3]]
        sub_mask = mask[:,:,sub_crop[0]:sub_crop[1],sub_crop[2]:sub_crop[3]]

        region_noise_pred, region_latents = sub_module(
            self=self,
            latents=sub_latents,
            t=t,
            module_kwargs=sub_module_kwargs,
            control_data=None,
            t2i_adapter_data=None,
            **kwargs,
        )
        threshhold = 1 - (t.item() / self.scheduler.config.num_train_timesteps)
        sub_mask_bool = sub_mask < threshhold

        total_noise_pred[:,:,sub_crop[0]:sub_crop[1],sub_crop[2]:sub_crop[3]] += torch.where(sub_mask_bool, region_noise_pred * region["weight"], torch.zeros_like(region_noise_pred))
        total_latents[:,:,sub_crop[0]:sub_crop[1],sub_crop[2]:sub_crop[3]] += torch.where(sub_mask_bool, region_latents * region["weight"], torch.zeros_like(region_latents))
        total_weight[:,:,sub_crop[0]:sub_crop[1],sub_crop[2]:sub_crop[3]] += torch.where(sub_mask_bool, torch.ones_like(region_noise_pred) * region["weight"], torch.zeros_like(region_noise_pred))

    total_noise_pred = total_noise_pred / total_weight
    total_latents = total_latents / total_weight

    return total_noise_pred, total_latents

@invocation("regional_noise_pred_module",
    title="Regional Noise Prediction Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class RegionalNoisePredModuleInvocation(BaseInvocation):
    """NP_MOD: Regional Noise Prediction"""
    default_sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the default noise prediction. No connection will use the default pipeline.",
        title="[NP] Default SubModule",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    region_info: Union[list[RegionalNoisePredictionData], RegionalNoisePredictionData] = InputField(
        default=[],
        description="The custom modules to use for each region. No connection will use the default pipeline.",
        title="[NP] Region Info",
        input=Input.Connection,
    )
    default_weight: float = InputField(
        title="Default Weight",
        description="The weight to apply to the default noise prediction",
        ge=0,
        default=1.0,
    )

    def invoke(self, context: InvocationContext) -> NP_ModuleDataOutput:
        if isinstance(self.region_info, RegionalNoisePredictionData):
            self.region_info = [self.region_info]

        module = NP_ModuleData(
            name="Regional Noise Prediction module",
            module="regional_noise_pred",
            module_kwargs={
                "default_sub_module": self.default_sub_module,
                "region_info": self.region_info,
                "default_weight": self.default_weight,
                "module_id": self.id,
            },
        )

        return NP_ModuleDataOutput(
            module_data_output=module,
        )
