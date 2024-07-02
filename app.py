import spaces
import torch
from diffusers import StableDiffusion3InstructPix2PixPipeline, SD3Transformer2DModel
import gradio as gr
import PIL.Image
import numpy as np
from PIL import Image, ImageOps


pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_w_mask", torch_dtype=torch.float16)

pipe = pipe.to("cuda")


def is_blank_mask(mask_img):
    # Convert the mask to a numpy array and check if all values are 0 (black/transparent)
    mask_array = np.array(mask_img.convert('L'))  # Convert to luminance to simplify the check
    return np.all(mask_array == 0)


def generate(image_mask, prompt, num_inference_steps=50, image_guidance_scale=1.6, guidance_scale=7.5, seed=255):
    # Set the seed for reproducibility
    generator = torch.manual_seed(seed)

    img = image_mask["image"]
    mask_img = image_mask["mask"]

    # Central crop to desired size
    desired_size = (512, 512)

    img = ImageOps.fit(img, desired_size, method=Image.LANCZOS, centering=(0.5, 0.5))
    mask_img = ImageOps.fit(mask_img, desired_size, method=Image.LANCZOS, centering=(0.5, 0.5))

    if is_blank_mask(mask_img):
        # Create a mask of the same size with all values set to 255 (white)
        mask_img = PIL.Image.new('RGB', img.size, color=(255, 255, 255))
    mask_img = mask_img.convert('RGB')

    image = pipe(
        prompt,
        image=img,
        mask_img=mask_img,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    return image


image_mask_input = gr.ImageMask(label="Input Image", type="pil", brush_color="#000000", elem_id="inputmask",
                                shape=(512, 512))
prompt_input = gr.Textbox(label="Prompt")
num_inference_steps_input = gr.Slider(minimum=0, maximum=100, value=50, label="Number of Inference Steps")
image_guidance_scale_input = gr.Slider(minimum=0.0, maximum=2.5, value=1.5, label="Image Guidance Scale")
guidance_scale_input = gr.Slider(minimum=0.0, maximum=17.5, value=12.5, label="Guidance Scale")
seed_input = gr.Textbox(value="255", label="Random Seed")

inputs = [image_mask_input, prompt_input, num_inference_steps_input, image_guidance_scale_input, guidance_scale_input,
          seed_input]
outputs = gr.Image(label="Generated Image")


def launch_interface():
    @spaces.GPU
    def generate_with_seed(image_mask, prompt, num_inference_steps, image_guidance_scale, guidance_scale, seed):
        # Convert seed to integer
        seed = int(seed)
        return generate(image_mask, prompt, num_inference_steps, image_guidance_scale, guidance_scale, seed)

    # Define inputs and outputs (replace these with your actual inputs and outputs)
    inputs = [
        gr.inputs.Image(type="numpy", label="Image Mask"),
        gr.inputs.Textbox(label="Prompt"),
        gr.inputs.Slider(minimum=1, maximum=100, step=1, label="Num Inference Steps"),
        gr.inputs.Slider(minimum=0.1, maximum=10.0, step=0.1, label="Image Guidance Scale"),
        gr.inputs.Slider(minimum=0.1, maximum=10.0, step=0.1, label="Guidance Scale"),
        gr.inputs.Textbox(label="Seed", default="1024")  # Set default value to 1024
    ]
    outputs = gr.outputs.Image(type="numpy", label="Generated Image")

    # Example data
    # examples = [
    #     ["path_to_image_mask1.png", "Example prompt 1", 50, 7.5, 7.5, "12345"],
    #     ["path_to_image_mask2.png", "Example prompt 2", 80, 8.0, 9.0, "67890"],
    #     # Add more examples as needed
    # ]

    # Custom HTML content
    article_html = """
    <h2>Welcome to the Image Generation Interface</h2>
    <p>This interface allows you to generate images based on a given mask and prompt. Use the sliders to adjust the inference steps and guidance scales, and provide a seed for reproducibility.</p>
    """

    demo = gr.Interface(
        fn=generate_with_seed,
        inputs=inputs,
        outputs=outputs,
        # examples=examples,  # Add examples parameter
        article=article_html  # Add article parameter
    )

    demo.queue().launch()


launch_interface()