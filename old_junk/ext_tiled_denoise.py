####################################################################################################
# MultiDiffusion Sampling
# From: https://multidiffusion.github.io/
####################################################################################################
from .denoise_latents_extensions import DenoiseExtensionSD12X, DenoiseLatentsData, guidance_extension_12X
from invokeai.backend.util.logging import info, warning, error
import torch
import torch.nn.functional as F
import random
from typing import Callable, Any, Literal
from invokeai.invocation_api import (
    invocation,
    BaseInvocation,
    InputField,
    InvocationContext,
)
from .denoise_latents_extensions import (
    GuidanceField,
    GuidanceDataOutput
)

MD_PAD_MODES = Literal[
    "constant",
    "reflect",
    "replicate",
]

@guidance_extension_12X("tiled_denoise")
class TiledDenoiseGuidance(DenoiseExtensionSD12X):
    """
    Splits the denoise process into multiple sub-tiles of the latent to reduce memory usage.
    """
    def list_modifies(self) -> dict[str, Callable[..., Any]]:
        return super().list_modifies() #REPLACE with {functionname: self.functionname, ...} if you have any modifies
    
    def list_swaps(self) -> dict[str, Callable[..., Any]]:
        return {
            "swap_do_unet_step": self.swap_do_unet_step,
        } #REPLACE with {functionname: self.functionname, ...} if you have any swaps
    
    def __post_init__(self, tile_size:int, stride: int, jitter: bool, pad_mode: MD_PAD_MODES):
        self.tile_size = tile_size
        self.stride = stride
        self.jitter = jitter
        self.pad_mode = pad_mode
    
    def _get_views(self, height, width, window_size=128, stride=64, random_jitter=False):
        info(f"Getting views for height: {height}, width: {width}, window_size: {window_size}, stride: {stride}, random_jitter: {random_jitter}")
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

    def swap_do_unet_step(
            self,
            default: Callable,
            sample: torch.Tensor,
            **kwargs
        ) -> tuple[torch.Tensor, torch.Tensor]:
        height = sample.shape[-2]
        width = sample.shape[-1]
        window_size = self.tile_size // 8
        stride = self.stride // 8

        views = self._get_views(
            height=height,
            width=width,
            window_size=window_size,
            stride=stride,
            random_jitter=self.jitter,
        )
        if self.jitter:
            jitter_range = (window_size - stride) // 4
            latents_pad = F.pad(sample, (jitter_range, jitter_range, jitter_range, jitter_range), self.pad_mode, 0)
        else:
            jitter_range = 0
            latents_pad = sample

        count_local_uc = torch.zeros_like(latents_pad)
        value_local_uc = torch.zeros_like(latents_pad)
        count_local_c = torch.zeros_like(latents_pad)
        value_local_c = torch.zeros_like(latents_pad)

        for j, view in enumerate(views):
            h_start, h_end, w_start, w_end = view
            latents_for_view = latents_pad[:, :, h_start:h_end, w_start:w_end]
        
            uc_noise_pred, c_noise_pred = default(sample=latents_for_view, **kwargs)
            count_local_uc[:, :, h_start:h_end, w_start:w_end] += 1
            value_local_uc[:, :, h_start:h_end, w_start:w_end] += uc_noise_pred
            count_local_c[:, :, h_start:h_end, w_start:w_end] += 1
            value_local_c[:, :, h_start:h_end, w_start:w_end] += c_noise_pred

        #crop the padding back off of each tensor
        if jitter_range > 0:
            count_local_uc = count_local_uc[:, :, jitter_range:-jitter_range, jitter_range:-jitter_range]
            value_local_uc = value_local_uc[:, :, jitter_range:-jitter_range, jitter_range:-jitter_range]
            count_local_c = count_local_c[:, :, jitter_range:-jitter_range, jitter_range:-jitter_range]
            value_local_c = value_local_c[:, :, jitter_range:-jitter_range, jitter_range:-jitter_range]

        uc_noise_pred = value_local_uc / count_local_uc
        c_noise_pred = value_local_c / count_local_c

        return uc_noise_pred, c_noise_pred

@invocation(
    "ext_tiled_denoise",
    title="EXT: Tiled Denoise",
    tags=["guidance", "extension", "tiled", "denoise"],
    category="guidance",
    version="1.0.0",
)
class EXT_TiledDenoiseGuidanceInvocation(BaseInvocation):
    """
    Reduces VRAM usage by splitting large images into smaller tiles during denoising.
    """
    priority: int = InputField(default=800, description="Priority of the guidance module", ui_order=0, ui_hidden=True)
    tile_size: int = InputField(default=512, ge=128, multiple_of=8, description="Size of each tile", ui_order=1)
    stride: int = InputField(default=256, ge=64, multiple_of=64, description="The distance from the start of each tile to the next", ui_order=2)
    apply_jitter: bool = InputField(default=False, description="Randomly shift the tiles to reduce visible seams. May require higher step counts.", ui_order=3)
    pad_mode: MD_PAD_MODES = InputField(default="reflect", description="Padding mode for the edges of the latent. Only used if jitter is True.", ui_order=4)

    def invoke(self, context: InvocationContext) -> GuidanceDataOutput:

        kwargs = dict(
            tile_size=self.tile_size,
            stride=self.stride,
            jitter=self.apply_jitter,
            pad_mode=self.pad_mode,
        )

        return GuidanceDataOutput(
                guidance_data_output=GuidanceField(
                    guidance_name="tiled_denoise", #matches the decorator name above the guidance class
                    priority=self.priority, #required by all guidance modules
                    extension_kwargs=kwargs, #custom keyword arguments for the guidance module, must be serializeable to JSON
                )
            )