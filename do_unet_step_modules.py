from .modular_noise_prediction import module_noise_pred, get_noise_prediction_module
from .modular_denoise_latents import Modular_StableDiffusionGeneratorPipeline, ModuleData, ModuleDataOutput
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData
import torch
from typing import Literal, Optional, Callable

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
    sample: torch.Tensor,
    t: torch.Tensor,
    conditioning_data: ConditioningData,  # TODO: type
    step_index: int,
    total_step_count: int,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
        # result from calling object's default pipeline
        uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
            sample=sample,
            timestep=t,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            **kwargs,
        )

        guidance_scale = conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[step_index]

        noise_pred = self.invokeai_diffuser._combine(
            uc_noise_pred,
            c_noise_pred,
            guidance_scale,
        )
        return noise_pred

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
# MultiDiffusion Sampling
# From: https://multidiffusion.github.io/
####################################################################################################
import random
import torch.nn.functional as F
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


@module_noise_pred("multidiffusion_sampling")
def multidiffusion_sampling(
    self: Modular_StableDiffusionGeneratorPipeline,
    sample: torch.Tensor,
    t: torch.Tensor,
    conditioning_data,  # TODO: type
    step_index: int,
    total_step_count: int,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    latent_model_input = sample
    height = latent_model_input.shape[2]
    width = latent_model_input.shape[3]
    window_size = module_kwargs["tile_size"] // 8
    stride = module_kwargs["stride"] // 8
    pad_mode = module_kwargs["pad_mode"]
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])

    views = get_views(height, width, stride=stride, window_size=window_size, random_jitter=True)

    jitter_range = (window_size - stride) // 4

    latents_ = F.pad(latent_model_input, (jitter_range, jitter_range, jitter_range, jitter_range), pad_mode, 0)

    count_local = torch.zeros_like(latents_)
    value_local = torch.zeros_like(latents_)
    
    for j, view in enumerate(views):
        h_start, h_end, w_start, w_end = view
        latents_for_view = latents_[:, :, h_start:h_end, w_start:w_end]

        noise_pred = sub_module(
            self=self,
            sample=latents_for_view,
            t=t,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            module_kwargs=sub_module_kwargs,
            **kwargs,
        )

        value_local[:, :, h_start:h_end, w_start:w_end] += noise_pred #step_output.prev_sample.detach().clone()
        count_local[:, :, h_start:h_end, w_start:w_end] += 1

    value_local_crop = value_local[: ,:, jitter_range: jitter_range + height, jitter_range: jitter_range + width]
    count_local_crop = count_local[: ,:, jitter_range: jitter_range + height, jitter_range: jitter_range + width]

    pred_multi = (value_local_crop / count_local_crop)

    return pred_multi


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
    """Module: MultiDiffusion tiled sampling"""
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
        multiple_of=8,
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
    sample: torch.Tensor,
    t: torch.Tensor,
    conditioning_data,  # TODO: type
    step_index: int,
    total_step_count: int,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    latent_model_input = sample
    gaussian_decay_rate = module_kwargs["gaussian_decay_rate"]
    dilation_scale = module_kwargs["dilation_scale"]
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (self.scheduler.config.num_train_timesteps - t) / self.scheduler.config.num_train_timesteps)).cpu()
    sigma = cosine_factor ** gaussian_decay_rate + 1e-2

    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])
        
    total_noise_pred = torch.zeros_like(latent_model_input)
    std_, mean_ = latent_model_input.std(), latent_model_input.mean()
    blurred_latents = gaussian_filter(latent_model_input, kernel_size=(2*dilation_scale-1), sigma=sigma)
    blurred_latents = (blurred_latents - blurred_latents.mean()) / blurred_latents.std() * std_ + mean_
    for h in range(dilation_scale):
        for w in range(dilation_scale):
            #get interlaced subsample
            subsample = blurred_latents[:, :, h::dilation_scale, w::dilation_scale]
            noise_pred = sub_module(
                self=self,
                sample=subsample,
                t=t,
                conditioning_data=conditioning_data,
                step_index=step_index,
                total_step_count=total_step_count,
                module_kwargs=sub_module_kwargs,
                **kwargs,
            )

            # insert subsample noise prediction into total tensor
            total_noise_pred[:, :, h::dilation_scale, w::dilation_scale] = noise_pred
    return total_noise_pred

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
        title="Dilation Scale",
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
    sample: torch.Tensor,
    t: torch.Tensor,
    conditioning_data,  # TODO: type
    step_index: int,
    total_step_count: int,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    decay_rate = module_kwargs["decay_rate"]
    sub_module_1, sub_module_1_kwargs = resolve_module(module_kwargs["sub_module_1"])
    sub_module_2, sub_module_2_kwargs = resolve_module(module_kwargs["sub_module_2"])

    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (self.scheduler.config.num_train_timesteps - t) / self.scheduler.config.num_train_timesteps)).cpu()
    c2 = 1 - cosine_factor ** decay_rate

    pred_1 = sub_module_1(
        self=self,
        sample=sample,
        t=t,
        conditioning_data=conditioning_data,
        step_index=step_index,
        total_step_count=total_step_count,
        module_kwargs=sub_module_1_kwargs,
        **kwargs,
    )

    pred_2 = sub_module_2(
        self=self,
        sample=sample,
        t=t,
        conditioning_data=conditioning_data,
        step_index=step_index,
        total_step_count=total_step_count,
        module_kwargs=sub_module_2_kwargs,
        **kwargs,
    )
    total_noise_pred = torch.lerp(pred_1, pred_2, c2.to(pred_1.device, dtype=pred_1.dtype))

    return total_noise_pred

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
    sample: torch.Tensor,
    t: torch.Tensor,
    conditioning_data,  # TODO: type
    step_index: int,
    total_step_count: int,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    start_step: int = module_kwargs["start_step"]
    end_step: int = module_kwargs["end_step"]
    sub_module_1, sub_module_1_kwargs = resolve_module(module_kwargs["sub_module_1"])
    sub_module_2, sub_module_2_kwargs = resolve_module(module_kwargs["sub_module_2"])
    
    linear_factor = (step_index - start_step) / (end_step - start_step)
    linear_factor = min(max(linear_factor, 0), 1)

    if linear_factor < 1:
        pred_1 = sub_module_1(
            self=self,
            sample=sample,
            t=t,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            module_kwargs=sub_module_1_kwargs,
            **kwargs,
        )

    if linear_factor > 0:
        pred_2 = sub_module_2(
            self=self,
            sample=sample,
            t=t,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            module_kwargs=sub_module_2_kwargs,
            **kwargs,
        )
    
    if linear_factor == 0:
        total_noise_pred = pred_1 # no need to lerp
        print(f"Linear Transfer: pred_1")
    elif linear_factor == 1:
        total_noise_pred = pred_2 # no need to lerp
        print(f"Linear Transfer: pred_2")
    else:
        total_noise_pred = torch.lerp(pred_1, pred_2, linear_factor)
        print(f"Linear Transfer: lerp(pred_1, pred_2, {linear_factor})")

    return total_noise_pred

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
@module_noise_pred("tiled_denoise")
def tiled_denoise_latents(
    self: Modular_StableDiffusionGeneratorPipeline,
    sample: torch.Tensor,
    t: torch.Tensor,
    conditioning_data,  # TODO: type
    step_index: int,
    total_step_count: int,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    latent_model_input = sample
    height = latent_model_input.shape[2]
    width = latent_model_input.shape[3]
    window_size = module_kwargs["tile_size"] // 8
    stride = max(window_size - (module_kwargs["overlap"] // 8),8)
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])

    views = get_views(height, width, stride=stride, window_size=window_size, random_jitter=False)

    count_local = torch.zeros_like(latent_model_input)
    value_local = torch.zeros_like(latent_model_input)
    
    for j, view in enumerate(views):
        h_start, h_end, w_start, w_end = view
        latents_for_view = latent_model_input[:, :, h_start:h_end, w_start:w_end]

        noise_pred = sub_module(
            self=self,
            sample=latents_for_view,
            t=t,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            module_kwargs=sub_module_kwargs,
            **kwargs,
        )

        value_local[:, :, h_start:h_end, w_start:w_end] += noise_pred #step_output.prev_sample.detach().clone()
        count_local[:, :, h_start:h_end, w_start:w_end] += 1

    pred_multi = (value_local / count_local)

    return pred_multi

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
            module="tiled_denoise",
            module_kwargs={
                "sub_module": self.sub_module,
                "tile_size": self.tile_size,
                "overlap": self.overlap,
            },
        )

        return ModuleDataOutput(
            module_data_output=module,
        )

####################################################################################################
# Color Correction
# From: https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
####################################################################################################

# Shrinking towards the mean (will also remove outliers)
def soft_clamp_tensor(input_tensor: torch.Tensor, threshold=0.9, boundary=4, channels=[0, 1, 2]):
    if max(abs(input_tensor.max()), abs(input_tensor.min())) < 4:
        return input_tensor
    for channel in channels:
        channel_tensor = input_tensor[:, channel, ...]

        max_val = channel_tensor.max()
        max_replace = ((channel_tensor - threshold) / (max_val - threshold)) * (boundary - threshold) + threshold
        over_mask = (channel_tensor > threshold)

        min_val = channel_tensor.min()
        min_replace = ((channel_tensor + threshold) / (min_val + threshold)) * (-boundary + threshold) - threshold
        under_mask = (channel_tensor < -threshold)

        input_tensor[:, channel, ...] = torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, channel_tensor))

    return input_tensor

# Center tensor (balance colors)
def center_tensor(input_tensor: torch.Tensor, channel_shift=1, full_shift=1, channels=[0, 1, 2, 3], center = 0):
    for channel in channels:
        input_tensor[0, channel] -= (input_tensor[0, channel].mean() - center) * channel_shift
    return input_tensor - (input_tensor.mean() - center) * full_shift

# Maximize/normalize tensor
def normalize_tensor(input_tensor, lower_bound, upper_bound, channels=[0, 1, 2], expand_dynamic_range=False):
    min_val = input_tensor.min()
    max_val = input_tensor.max()

    if expand_dynamic_range:
        normalization_factor = (upper_bound - lower_bound) / (max_val - min_val)
    else:
        normalization_factor = 1
    input_tensor[0, channels] = (input_tensor[0, channels] - min_val) * normalization_factor + lower_bound

    return input_tensor

@module_noise_pred("color_guidance")
def color_guidance(
    self: Modular_StableDiffusionGeneratorPipeline,
    sample: torch.Tensor,
    t: torch.Tensor,
    conditioning_data,  # TODO: type
    step_index: int,
    total_step_count: int,
    module_kwargs: dict | None,
    **kwargs,
) -> torch.Tensor:
    sub_module, sub_module_kwargs = resolve_module(module_kwargs["sub_module"])
    upper_bound: float = module_kwargs["upper_bound"]
    lower_bound: float = module_kwargs["lower_bound"]
    channels = module_kwargs["channels"]
    expand_dynamic_range: bool = module_kwargs["expand_dynamic_range"]
    timestep: float = t.item()

    noise_pred: torch.Tensor = sub_module(
        self=self,
        sample=sample,
        t=t,
        conditioning_data=conditioning_data,
        step_index=step_index,
        total_step_count=total_step_count,
        module_kwargs=sub_module_kwargs,
        **kwargs,
    )

    center = upper_bound * 0.5 + lower_bound * 0.5

    if timestep > 950:
        threshold = max(noise_pred.max(), abs(noise_pred.min())) * 0.998
        noise_pred = soft_clamp_tensor(noise_pred, threshold*0.998, threshold)
    if timestep > 700:
        noise_pred = center_tensor(noise_pred, 0.8, 0.8, channels=channels, center=center)
    if timestep > 1 and timestep < 100:
        noise_pred = center_tensor(noise_pred, 0.6, 1.0, channels=channels, center=center)
        noise_pred = normalize_tensor(noise_pred, lower_bound=lower_bound, upper_bound=upper_bound, channels=channels, expand_dynamic_range=expand_dynamic_range)

    return noise_pred

CHANNEL_SELECTIONS = Literal[
    "All Channels",
    "SDXL Colors Only",
    "L0: Brightness",
    "L1: SD1 Highlights // SDXL Red->Cyan",
    "L2: SD1 Red/Green // SDXL Magenta->Green",
    "L3: SD1 Magenta/Yellow // SDXL Structure",
]

CHANNEL_VALUES = {
    "All Channels": [0, 1, 2, 3],
    "SDXL Colors Only": [1, 2],
    "L0: Brightness": [0],
    "L1: SD1 Highlights // SDXL Red->Cyan": [1],
    "L2: SD1 Red/Green // SDXL Magenta->Green": [2],
    "L3: SD1 Magenta/Yellow // SDXL Structure": [3],
}

CHANNEL_DESCRIPTION = """The channels to affect in the latent correction.\n
SDXL: L1 = Red/Cyan, L2 = Magenta/Green, L3 = Structure\n
SD1.5: L1 = Shadows, L2 = Red/Green, L3 = Magenta/Yellow"""

@invocation("color_guidance_module",
    title="Color Guidance Module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
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
    adjustment: float = InputField(
        title="Adjustment",
        description="0: Will correct colors to remain within VAE bounds.\nOthervalues will shift the mean of the latent.\nRecommended range: -0.2->0.2",
        default=0,
    )
    channel_selection: CHANNEL_SELECTIONS = InputField(
        title="Channel Selection",
        description=CHANNEL_DESCRIPTION,
        default="All Channels",
        input=Input.Direct,
    )
    expand_dynamic_range: bool = InputField(
        title="Expand Dynamic Range",
        description="If true, will expand the dynamic range of the latent channels to match the range of the VAE.",
        default=True,
        input=Input.Direct,
    )

    def invoke(self, context: InvocationContext) -> ModuleDataOutput:

        channels = CHANNEL_VALUES[self.channel_selection]

        module = ModuleData(
            name="Color Guidance module",
            module_type="do_unet_step",
            module="color_guidance",
            module_kwargs={
                "sub_module": self.sub_module,
                "upper_bound": 4 + self.adjustment,
                "lower_bound": -4 + self.adjustment,
                "channels": channels,
                "expand_dynamic_range": self.expand_dynamic_range,
            },
        )

        return ModuleDataOutput(
            module_data_output=module,
        )