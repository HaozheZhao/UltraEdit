# For Editing with SD3
import torch
from diffusers import StableDiffusion3InstructPix2PixPipeline
from diffusers.utils import load_image
import requests
import PIL.Image
import PIL.ImageOps

pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_w_mask", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt="What if the horse wears a hat?"
img = load_image("input.png").resize((512, 512))
mask_img = load_image("mask_img.png").resize(img.size)
# For free form Editing, seed a blank mask
# mask_img = PIL.Image.new("RGB", img.size, (255, 255, 255))
image = pipe(
    prompt,
    image=img,
    mask_img=mask_img,
    negative_prompt="",
    num_inference_steps=50,
    image_guidance_scale=1.5,
    guidance_scale=7.5,
).images[0]
image.save("edited_image.png")
# display image


