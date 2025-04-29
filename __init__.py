from .analyse_latents import AnalyzeLatentsInvocation #extra for testing
from .exposed_denoise_latents import ExposedDenoiseLatentsInvocation
from .gradient_mask_extensions import GradientMaskExtensionInvocation
from .fam_extensions import FAM_FM_ExtensionInvocation, FAM_AM_ExtensionInvocation
from .refDrop_extensions import RefDrop_ExtensionInvocation
from .noise_investigation import (
    NoiseHeatmapInvocation,
    FourierLossCheckInvocation,
    CopyFrequencyValuesInvocation,
    ScheduledNoiseInvocation,
)
from .alt_CFG_extensions import TangentialDampingCFGExtensionInvocation
from .sw_guidance_extension import SWGuidanceExtensionInvocation
from .PLADIS_extension import PLADIS_ExtensionInvocation

"""
Running with SageAttention requires installing to the vent or compiling from the submodule. 
Navigate to SageAttention and run the following command:
pip install -e .

You will need CUDA 12.5+ for this to work. """
# from .sageAttention_extensions import SageAttention_ExtensionInvocation