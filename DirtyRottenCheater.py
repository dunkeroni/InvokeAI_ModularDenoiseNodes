from invokeai.app.invocations.baseinvocation import invocation
from invokeai.app.invocations.latent import DenoiseLatentsInvocation
from .DemoFusion_core import DemoFusionSDXLPipeline


@invocation(
    "dirty_rotten_cheater",
    title="Reroute Denoise to DemoFusion (dirty rotten cheater method)",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="1.0.0",
)
class DirtyRottenCheater(DenoiseLatentsInvocation):
    def create_pipeline(
        self,
        unet,
        scheduler,
    ) -> DemoFusionSDXLPipeline:
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

        return DemoFusionSDXLPipeline(
            vae=FakeVae(),  # TODO: oh...
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            unet=unet,
            scheduler=scheduler,
            #safety_checker=None,
            #feature_extractor=None,
            #requires_safety_checker=False,
        )