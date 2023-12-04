from invokeai.app.invocations.latent import DenoiseLatentsInvocation
from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline
from .DemoFusion_SDGP import DF_StableDiffusionGeneratorPipeline

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

@invocation(
    "demofusion_denoise_latents",
    title="DemoFusion Denoise Latents",
    tags=["demofusion", "generate"],
    category="demofusion",
    version="1.0.0",
)
class DemoFusionDenoiseLatentsInvocation(DenoiseLatentsInvocation):
    dilation_scale: int = InputField(
        title="Dilation Scale",
        description="Downsample rate for sampling. Should be res/base_res (e.g. '3' for SDXL at 3072x3072). 1 = scaling.",
        default=2,
        ge=1,
    )
    do_multi: bool = InputField(
        title="Enable Multi Diffusion",
        description="Whether to do multi-diffusion sampling",
        default=True,
    )
    do_dilation: bool = InputField(
        title="Enable Dilation",
        description="Whether to do dilation sampling",
        default=True,
    )
    multi_decay_rate: float = InputField(
        title="Multi Decay Rate",
        description="Decay rate Dilated Sampling => MultiDiffusion. Higher switches to multidiffusion faster.\nActive iff do_multi AND do_dilation",
        default=1,
        ge=0.0,
    )

    def create_pipeline(
        self,
        unet,
        scheduler,
    ) -> StableDiffusionGeneratorPipeline:
        # TODO:
        # configure_model_padding(
        #    unet,
        #    self.seamless,
        #    self.seamless_axes,
        # )

        class FakeVae:
            class FakeVaeConfig:
                def __init__(self):
                    self.block_out_channels = [0]

            def __init__(self):
                self.config = FakeVae.FakeVaeConfig()

        return DF_StableDiffusionGeneratorPipeline(
            vae=FakeVae(),  # TODO: oh...
            text_encoder=None,
            tokenizer=None,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            dilation_scale=self.dilation_scale,
            multi_decay_rate=self.multi_decay_rate,
            do_multi=self.do_multi,
            do_dilation=self.do_dilation,
        )