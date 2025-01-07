from invokeai.invocation_api import (
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    invocation,
    LatentsField,
    ImageField,
    ImageOutput,
)

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

@invocation("analyze_latents", title="Analyze Latents", tags=["analyze", "latents"], category="modular", version="1.0.0")
class AnalyzeLatentsInvocation(BaseInvocation):
    """ Create an image of a histogram of the latents with averages marked """
    latents: LatentsField = InputField(
        default=None, input=Input.Connection
    )
    bins: int = InputField(
        default=100,
        description="Number of bins to use in the histogram",
        title="Bins",
    )
    start_range: float = InputField(
        default=-4,
        input=Input.Direct,
        description="Start of the range to use in the histogram",
        title="Start Range",
    )
    end_range: float = InputField(
        default=4,
        input=Input.Direct,
        description="End of the range to use in the histogram",
        title="End Range",
    )
    image_title: str = InputField(
        default="Latent Histogram",
        input=Input.Direct,
        description="Title of the image",
        title="Image Title",
    )
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)
        latents = latents.detach().cpu().numpy()
        #split individual channels
        L0 = latents[0,0,:,:]
        L1 = latents[0,1,:,:]
        L2 = latents[0,2,:,:]
        L3 = latents[0,3,:,:]

        #create histogram
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].hist(L0.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
        axs[0, 0].set_title('L0')
        axs[0, 1].hist(L1.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
        axs[0, 1].set_title('L1')
        axs[1, 0].hist(L2.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
        axs[1, 0].set_title('L2')
        axs[1, 1].hist(L3.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
        axs[1, 1].set_title('L3')

        #add title
        fig.suptitle(self.image_title)

        plt.tight_layout()  # Adjust subplot spacing

        #add average lines
        axs[0, 0].axvline(x=L0.mean(), color='r', linestyle='dashed', linewidth=1)
        axs[0, 1].axvline(x=L1.mean(), color='r', linestyle='dashed', linewidth=1)
        axs[1, 0].axvline(x=L2.mean(), color='r', linestyle='dashed', linewidth=1)
        axs[1, 1].axvline(x=L3.mean(), color='r', linestyle='dashed', linewidth=1)

        #conver to PIL image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)
        img = Image.fromarray(buf)

        #return image
        image_dto = context.images.save(image=img)

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=img.width,
            height=img.height,
        )