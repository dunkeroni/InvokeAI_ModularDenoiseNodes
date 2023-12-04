import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

from .DemoFusion_gitclone.pipeline_demofusion_sdxl import DemoFusionSDXLPipeline

import torch

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.primitives import BoardField, ColorField, ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecordChanges, ResourceOrigin

@invocation(
    "demofusion_raw",
    title="DemoFusion RAW code activator",
    tags=["demofusion", "generate"],
    category="demofusion",
    version="1.0.0",
)
class DemoFusionRawInvocation(BaseInvocation):
    def invoke(self, context: InvocationContext) -> ImageOutput:

        def image_grid(imgs, save_path=None):

            w = 0
            for i, img in enumerate(imgs):
                h_, w_ = imgs[i].size
                w += w_
            h = h_
            grid = Image.new('RGB', size=(w, h))
            grid_w, grid_h = grid.size

            w = 0
            for i, img in enumerate(imgs):
                h_, w_ = imgs[i].size
                grid.paste(img, box=(w, h - h_))
                if save_path != None:
                    img.save(save_path + "/img_{}.jpg".format((i + 1) * 1024))
                w += w_
                
            return grid


        model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)

        pipe = pipe.to("cuda")

        prompt = "a dog on a log."
        negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

        generator = torch.Generator(device='cuda')
        generator = generator.manual_seed(522)

        images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                    height=2048, width=2048, view_batch_size=16, stride=64,
                    num_inference_steps=50, guidance_scale = 7.5,
                    cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8, 
                    multi_decoder=True, show_image=True
                    )
        num_images = len(images)

        #image_grid(images, save_path="./outputs/")

        image_dto = context.services.images.create(
            image=images[num_images-1],
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=None,
            workflow=None,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )