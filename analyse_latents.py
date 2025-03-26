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
        latents = context.tensors.load(self.latents.latents_name).half()
        latents = latents.detach().cpu().numpy()
        
        num_channels = latents.shape[1]
        if num_channels not in [4, 16]:
            raise ValueError("Latents must have either 4 or 16 channels")
        
        # Create subplots based on the number of channels
        if num_channels == 16:
            fig, axs = plt.subplots(4, 4, figsize=(15, 15))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()
        
        for i in range(num_channels):
            channel_data = latents[0, i, :, :]
            axs[i].hist(channel_data.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
            axs[i].set_title(f'L{i}')
            axs[i].axvline(x=channel_data.mean(), color='r', linestyle='dashed', linewidth=1)
        
        # Add title
        fig.suptitle(self.image_title)
        plt.tight_layout()  # Adjust subplot spacing
        
        # Convert to PIL image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)
        buf = np.roll(buf, 3, axis=2)  # Convert ARGB to RGBA
        img = Image.fromarray(buf)
        
        # Resize image if there are 16 channels
        if num_channels == 16:
            img = img.resize((1024, 1024))
        
        # Return image
        image_dto = context.images.save(image=img)
        
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=img.width,
            height=img.height,
        )

@invocation("analyze_image", title="Analyze Image", tags=["analyze", "image"], category="modular", version="1.1.0")
class AnalyzeImageInvocation(BaseInvocation):
    """ Create an image of a histogram of the image with averages marked """
    image: ImageField = InputField(
        default=None, input=Input.Connection
    )
    bins: int = InputField(
        default=255,
        description="Number of bins to use in the histogram",
        title="Bins",
    )
    start_range: float = InputField(
        default=0,
        input=Input.Direct,
        description="Start of the range to use in the histogram",
        title="Start Range",
    )
    end_range: float = InputField(
        default=255,
        input=Input.Direct,
        description="End of the range to use in the histogram",
        title="End Range",
    )
    image_title: str = InputField(
        default="Image Histogram",
        input=Input.Direct,
        description="Title of the image",
        title="Image Title",
    )
    greyscale_mode: bool = InputField(
        default=False,
        input=Input.Direct,
        description="Load image in greyscale mode",
        title="Greyscale Mode",
    )
    def invoke(self, context: InvocationContext) -> ImageOutput:
        if self.greyscale_mode:
            image = context.images.get_pil(self.image.image_name, mode="L")
            image = np.array(image)
            
            #create histogram for greyscale image
            fig, ax = plt.subplots()
            ax.hist(image.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
            ax.set_title('Greyscale Channel')
            
            #add title
            fig.suptitle(self.image_title)
            
            plt.tight_layout()  # Adjust subplot spacing
            
            #add average line if within range
            if self.start_range <= image.mean() <= self.end_range:
                ax.axvline(x=image.mean(), color='r', linestyle='dashed', linewidth=1)
            
            #convert to PIL image
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (h, w, 4)
            buf = np.roll(buf, 3, axis=2)  # Convert ARGB to RGBA
            img = Image.fromarray(buf)
        else:
            image = context.images.get_pil(self.image.image_name, mode="RGBA")
            image = np.array(image)
    
            #split individual channels
            R = image[:,:,0]
            G = image[:,:,1]
            B = image[:,:,2]
            A = image[:,:,3]
    
            #create histogram
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].hist(R.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
            axs[0, 0].set_title('Red Channel')
            axs[0, 1].hist(G.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
            axs[0, 1].set_title('Green Channel')
            axs[1, 0].hist(B.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
            axs[1, 0].set_title('Blue Channel')
            axs[1, 1].hist(A.flatten(), bins=self.bins, range=(self.start_range, self.end_range))
            axs[1, 1].set_title('Alpha Channel')
    
            #add title
            fig.suptitle(self.image_title)
    
            plt.tight_layout()  # Adjust subplot spacing
    
            #add average lines if within range
            if self.start_range <= R.mean() <= self.end_range:
                axs[0, 0].axvline(x=R.mean(), color='r', linestyle='dashed', linewidth=1)
            if self.start_range <= G.mean() <= self.end_range:
                axs[0, 1].axvline(x=G.mean(), color='r', linestyle='dashed', linewidth=1)
            if self.start_range <= B.mean() <= self.end_range:
                axs[1, 0].axvline(x=B.mean(), color='r', linestyle='dashed', linewidth=1)
            if self.start_range <= A.mean() <= self.end_range:
                axs[1, 1].axvline(x=A.mean(), color='r', linestyle='dashed', linewidth=1)
    
            #convert to PIL image
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (h, w, 4)
            buf = np.roll(buf, 3, axis=2)  # Convert ARGB to RGBA
            img = Image.fromarray(buf)
    
        #return image
        image_dto = context.images.save(image=img)
    
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=img.width,
            height=img.height,
        )

@invocation("image_difference", title="Image Difference", tags=["analyze", "image"], category="modular", version="1.0.0")
class ImageDifferenceInvocation(BaseInvocation):
    """ Create an image of the difference between two images """
    image1: ImageField = InputField(
        default=None, input=Input.Connection
    )
    image2: ImageField = InputField(
        default=None, input=Input.Connection
    )
    image_title: str = InputField(
        default="Image Difference",
        input=Input.Direct,
        description="Title of the image",
        title="Image Title",
    )
    def invoke(self, context: InvocationContext) -> ImageOutput:
        image1 = context.images.get_pil(self.image1.image_name, mode="RGBA")
        image2 = context.images.get_pil(self.image2.image_name, mode="RGBA")
        image1 = np.array(image1)
        image2 = np.array(image2)
        
        #check if images are the same size
        if image1.shape != image2.shape:
            raise ValueError("Images must be the same size")
        
        #calculate difference
        diff = np.clip((image1 - image2) + 127, 0, 255).astype(np.uint8)
        
        #convert to PIL image
        img = Image.fromarray(diff)
        
        #return image
        image_dto = context.images.save(image=img)
        
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=img.width,
            height=img.height,
        )