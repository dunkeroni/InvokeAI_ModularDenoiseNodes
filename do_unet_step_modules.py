from .modular_noise_prediction import module_noise_pred, get_noise_prediction_module
from .modular_denoise_latents import Modular_StableDiffusionGeneratorPipeline, ModuleData, ModuleDataOutput
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData
import torch
from typing import Literal, Optional

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

####################################################################################################
# Standard UNet Step Module
####################################################################################################
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
    if module_kwargs["sub_module"] is None:
        sub_module = get_noise_prediction_module(None) #default case
        sub_module_kwargs = {}
    else:
        sub_module = get_noise_prediction_module(module_kwargs["sub_module"]["module"])
        sub_module_kwargs = module_kwargs["sub_module"]["module_kwargs"]

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

    if module_kwargs["sub_module"] is None:
        sub_module = get_noise_prediction_module(None) #default case
        sub_module_kwargs = {}
    else:
        sub_module = get_noise_prediction_module(module_kwargs["sub_module"]["module"])
        sub_module_kwargs = module_kwargs["sub_module"]["module_kwargs"]
        
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
    title="Dilated Sampling module",
    tags=["module", "modular"],
    category="modular",
    version="1.0.0",
)
class DilatedSamplingModuleInvocation(BaseInvocation):
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
    if module_kwargs["sub_module_1"] is None:
        sub_module_1 = get_noise_prediction_module(None) #default case
        sub_module_1_kwargs = {}
    else:
        sub_module_1 = get_noise_prediction_module(module_kwargs["sub_module_1"]["module"])
        sub_module_1_kwargs = module_kwargs["sub_module_1"]["module_kwargs"]
    
    if module_kwargs["sub_module_2"] is None:
        sub_module_2 = get_noise_prediction_module(None)
        sub_module_2_kwargs = {}
    else:
        sub_module_2 = get_noise_prediction_module(module_kwargs["sub_module_2"]["module"])
        sub_module_2_kwargs = module_kwargs["sub_module_2"]["module_kwargs"]

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
    sub_module_1: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the first noise prediction. No connection will use the default pipeline.",
        title="SubModules",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    sub_module_2: Optional[ModuleData] = InputField(
        default=None,
        description="The custom module to use for the second noise prediction. No connection will use the default pipeline.",
        title="SubModules",
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
