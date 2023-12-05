import math
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
import random
from invokeai.backend.ip_adapter.unet_patcher import UNetPatcher
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData

from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    IPAdapterData,
    T2IAdapterData
)

from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline

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

class DF_StableDiffusionGeneratorPipeline(StableDiffusionGeneratorPipeline):
    """
    DemoFusion Stable Diffusion Generator Pipeline
    Overrides the StableDiffusionGeneratorPipeline class from the Stable Diffusion package.
    Allows integration with the existing Denoise Latents node architecture
    """
    def __init__(self, *args, **kwargs):
        self.dilation_scale = kwargs.pop("dilation_scale")
        self.multi_decay_rate = kwargs.pop("multi_decay_rate")
        self.do_multi = kwargs.pop("do_multi")
        self.do_dilation = kwargs.pop("do_dilation")
        super().__init__(*args, **kwargs)
        # self.dilation_scale = dilation_scale
        # self.dilated_decay_rate = dilated_decay_rate
        # self.multi_decay_rate = multi_decay_rate
    

    def get_views(self, height, width, window_size=128, stride=64, random_jitter=False):
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


    @torch.inference_mode()
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


        ######################################  DemoFusion Modifications ######################################
        # DemoFusion combines three techniques to improve the quality of denoising high resolution images:
        # 1. Skip Residual (NOT IMPLEMENTED): use a synthetically noised version of the input latent to guide towards the original non-noised result
        # 2. Multi-Sampling: break the global latent into smaller tiles, and sample each tile separately
        # 3. Dilated Sampling: sample the latent at a lower resolution, and interlace the results to the original resolution


        # 0. prepare collector values
        count = torch.zeros_like(latent_model_input)
        value = torch.zeros_like(latent_model_input)
        height = latent_model_input.shape[2]
        width = latent_model_input.shape[3]
        cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (self.scheduler.config.num_train_timesteps - t) / self.scheduler.config.num_train_timesteps)).cpu()
        do_multi_diffusion = self.do_multi
        do_dilated_sampling = self.do_dilation # and timestep > threshhold?
        c2 = cosine_factor ** self.multi_decay_rate


        # 1. apply skip residual
        # Requires access to the input latent of the entire process and the noise input to create a skip residual
        # Not implemented here since the pipeline.step() cannot gain access to the original input latent


        # 2. apply multi-sampling
        # multi-sampling splits the tensor into smaller tiles, and samples each tile separately
        # this is done to reduce memory usage
        if do_multi_diffusion:
            if height > width:
                window_size = height // self.dilation_scale
            else:
                window_size = width // self.dilation_scale
            stride = window_size // 2
            views = self.get_views(height, width, stride=stride, window_size=window_size, random_jitter=True)

            jitter_range = (window_size - stride) // 4

            latents_ = F.pad(latent_model_input, (jitter_range, jitter_range, jitter_range, jitter_range), 'constant', 0)

            count_local = torch.zeros_like(latents_)
            value_local = torch.zeros_like(latents_)
            
            for j, view in enumerate(views):
                h_start, h_end, w_start, w_end = view
                latents_for_view = latents_[:, :, h_start:h_end, w_start:w_end]

                uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
                    sample=latents_for_view,
                    timestep=t,  # TODO: debug how handled batched and non batched timesteps
                    step_index=step_index,
                    total_step_count=total_step_count,
                    conditioning_data=conditioning_data,
                    # extra:
                    #down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
                    #mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
                    #down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
                )

                guidance_scale = conditioning_data.guidance_scale
                if isinstance(guidance_scale, list):
                    guidance_scale = guidance_scale[step_index]

                noise_pred = self.invokeai_diffuser._combine(
                    uc_noise_pred,
                    c_noise_pred,
                    guidance_scale,
                )

                # compute the previous noisy sample x_t -> x_t-1
                #self.scheduler._init_step_index(timestep)
                #step_output = self.scheduler.step(noise_pred, timestep, latents_for_view, **conditioning_data.scheduler_args)

                value_local[:, :, h_start:h_end, w_start:w_end] += noise_pred #step_output.prev_sample.detach().clone()
                count_local[:, :, h_start:h_end, w_start:w_end] += 1

            value_local_crop = value_local[: ,:, jitter_range: jitter_range + height, jitter_range: jitter_range + width]
            count_local_crop = count_local[: ,:, jitter_range: jitter_range + height, jitter_range: jitter_range + width]
            
            #c2 = cosine_factor ** self.multi_decay_rate

            pred_multi = (value_local_crop / count_local_crop) # * (1 - c2)
            #count_multi = torch.ones_like(value_local)# * (1 - c2)


        # 3. apply dilated noise sampling
        # dilated sampling splits the tesor into interlaced blocks, and samples each block separately
        # this is done to reduce memory usage
        if do_dilated_sampling:
            total_noise_pred = torch.zeros_like(latent_model_input)
            sigma = cosine_factor ** self.multi_decay_rate + 1e-2 #maybe should pass separate rate, but this is fine for now
            std_, mean_ = latent_model_input.std(), latent_model_input.mean()
            blurred_latents = gaussian_filter(latent_model_input, kernel_size=(2*self.dilation_scale-1), sigma=sigma)
            blurred_latents = (blurred_latents - blurred_latents.mean()) / blurred_latents.std() * std_ + mean_
            for h in range(self.dilation_scale):
                for w in range(self.dilation_scale):
                    #get interlaced subsample
                    subsample = blurred_latents[:, :, h::self.dilation_scale, w::self.dilation_scale]
                    uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
                        sample=subsample,
                        timestep=t,  # TODO: debug how handled batched and non batched timesteps
                        step_index=step_index,
                        total_step_count=total_step_count,
                        conditioning_data=conditioning_data,
                        # extra:
                        down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
                        mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
                        down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
                    )

                    guidance_scale = conditioning_data.guidance_scale
                    if isinstance(guidance_scale, list):
                        guidance_scale = guidance_scale[step_index]

                    noise_pred = self.invokeai_diffuser._combine(
                        uc_noise_pred,
                        c_noise_pred,
                        guidance_scale,
                    )

                    # sinsert subsample noise prediction into total tensor
                    total_noise_pred[:, :, h::self.dilation_scale, w::self.dilation_scale] = noise_pred

            # std_, mean_ = total_noise_pred.std(), total_noise_pred.mean()
            # noise_pred_gaussian = gaussian_filter(total_noise_pred, kernel_size=(2*self.dilation_scale-1), sigma=1)
            # noise_pred_gaussian = (noise_pred_gaussian - noise_pred_gaussian.mean()) / noise_pred_gaussian.std() * std_ + mean_

            
        #step_output.prev_sample = torch.where(count > 0.5, value / count, value).detach().clone().to(latents.device)
        if do_multi_diffusion and do_dilated_sampling:
            total_noise_pred = torch.lerp(pred_multi, total_noise_pred, c2.to(total_noise_pred.device, dtype=total_noise_pred.dtype))
        elif do_multi_diffusion:
            total_noise_pred = pred_multi
        elif do_dilated_sampling:
            total_noise_pred = total_noise_pred
        else: #normal diffusion pipeline
            uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
                sample=latent_model_input,
                timestep=t,  # TODO: debug how handled batched and non batched timesteps
                step_index=step_index,
                total_step_count=total_step_count,
                conditioning_data=conditioning_data,
                # extra:
                down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
                mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
            )

            guidance_scale = conditioning_data.guidance_scale
            if isinstance(guidance_scale, list):
                guidance_scale = guidance_scale[step_index]

            total_noise_pred = self.invokeai_diffuser._combine(
                uc_noise_pred,
                c_noise_pred,
                guidance_scale,
            )

        # compute the previous noisy sample x_t -> x_t-1
        #self.scheduler._init_step_index(t) # I don't know why this is needed, but it stops things from breaking
        step_output = self.scheduler.step(total_noise_pred, timestep, latents, **conditioning_data.scheduler_args)

        #c2 = cosine_factor ** self.multi_decay_rate
        #value_dilated = step_output.prev_sample.to("cpu", dtype = latent_model_input.dtype)   # * c2
        #count_dilated = torch.ones_like(step_output.prev_sample).to("cpu")# * c2

        # compute the final sample x_t -> x_t as a combination of the two sampling methods

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