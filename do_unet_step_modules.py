from .modular_decorators import module_noise_pred, get_noise_prediction_module
from .modular_denoise_latents import Modular_StableDiffusionGeneratorPipeline, ModuleData, ModuleDataOutput

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData
from invokeai.backend.stable_diffusion.diffusers_pipeline import ControlNetData, T2IAdapterData
from invokeai.backend.stable_diffusion.diffusion.shared_invokeai_diffusion import PostprocessingSettings
from invokeai.app.invocations.primitives import LatentsField
from invokeai.app.invocations.compel import ConditioningField
import torch
from typing import Literal, Optional, Callable, List
import random
import torch.nn.functional as F

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    UIType,
    invocation,
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
    """Module: InvokeAI standard noise prediction."""
    def invoke(self, context: InvocationContext) -> ModuleDataOutput:
        module = ModuleData(
            name="Standard UNet Step Module",
            module_type="do_unet_step",
            module="standard_unet_step_module",
            module_kwargs={},
        )

        return ModuleDataOutput(
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
            unconditional_name: ConditioningField = module_kwargs["unconditional_name"]
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

        return noise_pred, latents

@invocation("perp_neg_unet_step_module",
    title="Perp Negative",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class PerpNegStepModuleInvocation(BaseInvocation):
    """Module: Perp Negative noise prediction."""
    unconditional: ConditioningField = InputField(
        description="EMPTY CONDITIONING GOES HERE",
        input=Input.Connection, 
    )
    def invoke(self, context: InvocationContext) -> ModuleDataOutput:
        module = ModuleData(
            name="Perp Negative",
            module_type="do_unet_step",
            module="perp_neg_unet_step",
            module_kwargs={
                "unconditional_name": self.unconditional.conditioning_name,
                "module_id": self.id,
            },
        )

        return ModuleDataOutput(
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
    """Module: MultiDiffusion tiled sampling. NOT compatible with t2i adapters."""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="SubModules",
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

    def invoke(self, context: InvocationContext) -> ModuleDataOutput:
        module = ModuleData(
            name="MultiDiffusion Sampling Step module",
            module_type="do_unet_step",
            module="multidiffusion_sampling",
            module_kwargs={
                "sub_module": self.sub_module,
                "tile_size": self.tile_size,
                "stride": self.stride,
                "pad_mode": self.pad_mode,
                "enable_jitter": True,
            },
        )

        return ModuleDataOutput(
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
    """Module: Dilated Sampling"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each interlaced noise prediction. No connection will use the default pipeline.",
        title="SubModules",
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

    def invoke(self, context: InvocationContext) -> ModuleDataOutput:
        module = ModuleData(
            name="Dilated Sampling Step module",
            module_type="do_unet_step",
            module="dilated_sampling",
            module_kwargs={
                "sub_module": self.sub_module,
                "dilation_scale": self.dilation_scale,
                "gaussian_decay_rate": self.gaussian_decay_rate,
            },
        )

        return ModuleDataOutput(
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
    """Module: Smoothly changed modules based on remaining denoise"""
    sub_module_1: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the first noise prediction. No connection will use the default pipeline.",
        title="SubModule 1",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    sub_module_2: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the second noise prediction. No connection will use the default pipeline.",
        title="SubModule 2",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    decay_rate: float = InputField(
        title="Cosine Decay Rate",
        description="The decay rate to use when combining the two noise predictions. Higher values will shift the balance towards the second noise prediction sooner",
        ge=0,
        default=1,
    )

    def invoke(self, context: InvocationContext) -> ModuleDataOutput:
        module = ModuleData(
            name="Cosine Decay Transfer module",
            module_type="do_unet_step",
            module="cosine_decay_transfer",
            module_kwargs={
                "sub_module_1": self.sub_module_1,
                "sub_module_2": self.sub_module_2,
                "decay_rate": self.decay_rate,
            },
        )

        return ModuleDataOutput(
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
    """Module: Smoothly change modules based on step."""
    sub_module_1: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the first noise prediction. No connection will use the default pipeline.",
        title="SubModule 1",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    sub_module_2: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the second noise prediction. No connection will use the default pipeline.",
        title="SubModule 2",
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

    def invoke(self, context: InvocationContext) -> ModuleDataOutput:
        module = ModuleData(
            name="Linear Transfer module",
            module_type="do_unet_step",
            module="linear_transfer",
            module_kwargs={
                "sub_module_1": self.sub_module_1,
                "sub_module_2": self.sub_module_2,
                "start_step": self.start_step,
                "end_step": self.end_step,
            },
        )

        return ModuleDataOutput(
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
    """Module: Denoise latents using tiled noise prediction"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="SubModules",
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

    def invoke(self, context: InvocationContext) -> ModuleDataOutput:
        module = ModuleData(
            name="Tiled Denoise Latents module",
            module_type="do_unet_step",
            module="multidiffusion_sampling",
            module_kwargs={
                "sub_module": self.sub_module,
                "tile_size": self.tile_size,
                "stride": self.tile_size - self.overlap,
                "enable_jitter": False,
                "pad_mode": None,
            },
        )

        return ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# SDXL Color Guidance
# From: https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
####################################################################################################

# Shrinking towards the mean (will also remove outliers)
def soft_clamp_tensor(input_tensor: torch.Tensor, threshold=0.9, boundary=4, channels=[0, 1, 2]):
    for channel in channels:
        channel_tensor = input_tensor[:, channel]
        if not max(abs(channel_tensor.max()), abs(channel_tensor.min())) < 4:
            max_val = channel_tensor.max()
            max_replace = ((channel_tensor - threshold) / (max_val - threshold)) * (boundary - threshold) + threshold
            over_mask = (channel_tensor > threshold)

            min_val = channel_tensor.min()
            min_replace = ((channel_tensor + threshold) / (min_val + threshold)) * (-boundary + threshold) - threshold
            under_mask = (channel_tensor < -threshold)

            input_tensor[:, channel] = torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, channel_tensor))

    return input_tensor

# Center tensor (balance colors)
def shift_tensor(input_tensor, channel_shift=1, channels=[0, 1, 2, 3], target = 0):
    for channel in channels:
        input_tensor[0, channel] -= (input_tensor[0, channel].mean() - target) * channel_shift
    return input_tensor# - input_tensor.mean() * full_shift

# Maximize/normalize tensor
def expand_tensor(input_tensor, boundary=4, channels=[0, 1, 2]):
    for channel in channels:
        input_tensor[0, channel] *= (boundary/2) / input_tensor[0, channel].max()
        #min_val = input_tensor[0, channel].min()
        #max_val = input_tensor[0, channel].max()

        #get min max from 3 standard deviations from mean instead
        mean = input_tensor[0, channel].mean()
        std = input_tensor[0, channel].std()
        min_val = mean - std * 2
        max_val = mean + std * 2

        #colors will always center around 0 for SDXL latents, but brightness/structure will not. Need to adjust this.
        normalization_factor = boundary / max(abs(min_val), abs(max_val))
        input_tensor[0, channel] *= normalization_factor

    return input_tensor

@module_noise_pred("color_guidance")
def color_guidance(
    self: Modular_StableDiffusionGeneratorPipeline,
    latents: torch.Tensor,
    step_index: int,
    t: torch.Tensor,
    module_kwargs: dict | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])
    expand_dynamic_range = False #module_kwargs["expand_dynamic_range"]
    #dynamic_range = module_kwargs["dynamic_range"]
    start_step = module_kwargs["start_step"]
    end_step = module_kwargs["end_step"]
    target_mean = module_kwargs["target_mean"]
    channels = module_kwargs["channels"]
    # expand_dynamic_range: bool = module_kwargs["expand_dynamic_range"]
    timestep: float = t.item()

    if step_index >= start_step and (step_index <= end_step or end_step == -1):
        latents = shift_tensor(latents, 1, channels=channels, target=target_mean)
        # if expand_dynamic_range:
        #     latents = expand_tensor(latents, boundary=dynamic_range, channels=channels)
    
    noise_pred, original_latents = sub_module(
        self=self,
        latents=latents,
        t=t,
        step_index=step_index,
        module_kwargs=sub_module_kwargs,
        **kwargs,
    )

    return noise_pred, original_latents

CHANNEL_SELECTIONS = Literal[
    "All Channels",
    "Colors Only",
    "L0: Brightness",
    "L1: Red->Cyan",
    "L2: Lime->Purple",
    "L3: Structure",
]

CHANNEL_VALUES = {
    "All Channels": [0, 1, 2, 3],
    "Colors Only": [1, 2],
    "L0: Brightness": [0],
    "L1: Cyan->Red": [1],
    "L2: Lime->Purple": [2],
    "L3: Structure": [3],
}

@invocation("color_guidance_module",
    title="Color Guidance Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.1",
)
class ColorGuidanceModuleInvocation(BaseInvocation):
    """Module: Color Guidance (fix SDXL yellow bias)"""
    sub_module: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for each noise prediction tile. No connection will use the default pipeline.",
        title="SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    start_step: int = InputField(
        title="Start Step",
        description="The step index at which to start applying color correction",
        ge=0,
        default=0,
    )
    end_step: int = InputField(
        title="End Step",
        description="The step index at which to stop applying color correction. -1 to never stop.",
        ge=-1,
        default=-1,
    )
    channel_selection: CHANNEL_SELECTIONS = InputField(
        title="Channel Selection",
        description="The channels to affect in the latent correction",
        default="All Channels",
        input=Input.Direct,
    )
    target_mean: float = InputField(
        title="Target Mean",
        description="The target mean to use for the latent correction",
        default=0,
    )

    def invoke(self, context: InvocationContext) -> ModuleDataOutput:

        channels = CHANNEL_VALUES[self.channel_selection]

        module = ModuleData(
            name="Color Guidance module",
            module_type="do_unet_step",
            module="color_guidance",
            module_kwargs={
                "sub_module": self.sub_module,
                "start_step": self.start_step,
                "end_step": self.end_step,
                "target_mean": self.target_mean,
                "channels": channels,
            },
        )

        return ModuleDataOutput(
            module_data_output=module,
        )


####################################################################################################
# Skip Residual 
# From: https://ruoyidu.github.io/demofusion/demofusion.html
####################################################################################################
import time
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
    """Module: Skip Residual"""
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

    def invoke(self, context: InvocationContext) -> ModuleDataOutput:
        module = ModuleData(
            name="Skip Residual module",
            module_type="do_unet_step",
            module="skip_residual",
            module_kwargs={
                "latent_input": self.latent_input,
                "noise_input": self.noise_input,
                "module_id": self.id,
            },
        )

        return ModuleDataOutput(
            module_data_output=module,
        )