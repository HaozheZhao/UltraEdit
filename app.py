import spaces
import torch
from diffusers import StableDiffusion3InstructPix2PixPipeline, SD3Transformer2DModel
import gradio as gr
import PIL.Image
import numpy as np
from PIL import Image, ImageOps


pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_w_mask", torch_dtype=torch.float16)

pipe = pipe.to("cuda")



@spaces.GPU(duration=20)
def generate(image_mask, prompt, num_inference_steps=50, image_guidance_scale=1.6, guidance_scale=7.5, seed=255):
    def is_blank_mask(mask_img):
        # Convert the mask to a numpy array and check if all values are 0 (black/transparent)
        mask_array = np.array(mask_img.convert('L'))  # Convert to luminance to simplify the check
        return np.all(mask_array == 0)
    # Set the seed for reproducibility
    seed = int(seed)
    generator = torch.manual_seed(seed)

    img = image_mask["background"].convert("RGB")
    mask_img = image_mask["layers"][0].getchannel('A').convert("RGB")

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

example_lists=[
    
    [['UltraEdit/images/example_images/1-input.png','UltraEdit/images/example_images/1-mask.png','UltraEdit/images/example_images/1-merged.png'], "Add a moon in the sky", 20, 1.5, 12.5,255],
    
    [['UltraEdit/images/example_images/1-input.png','UltraEdit/images/example_images/1-input.png','UltraEdit/images/example_images/1-input.png'], "Add a moon in the sky", 20, 1.5, 6.5,255],
    
    [['UltraEdit/images/example_images/2-input.png','UltraEdit/images/example_images/2-mask.png','UltraEdit/images/example_images/2-merged.png'], "add cherry blossoms", 20, 1.5, 12.5,255],
    
    [['UltraEdit/images/example_images/3-input.png','UltraEdit/images/example_images/3-mask.png','UltraEdit/images/example_images/3-merged.png'], "Please dress her in a short purple wedding dress adorned with white floral embroidery.", 20, 1.5, 6.5,255],

    [['UltraEdit/images/example_images/4-input.png','UltraEdit/images/example_images/4-mask.png','UltraEdit/images/example_images/4-merged.png'], "give her a chief's headdress.", 20, 1.5, 7.5, 24555]

]
mask_ex_list = []
for exp in example_lists:
    ex_dict= {}
    ex_dict['background'] = exp[0][0]
    ex_dict['layers'] =  [exp[0][1],exp[0][2]]
    ex_dict['composite'] =  exp[0][2]
    re_list = [ex_dict, exp[1],exp[2],exp[3],exp[4],exp[5]]
    mask_ex_list.append(re_list)

# image_mask_input = gr.ImageMask(label="Input Image", type="pil", brush_color="#000000", elem_id="inputmask",
#                                 shape=(512, 512))
image_mask_input = gr.ImageMask(sources='upload',type="pil",label="Input Image: Mask with pen or leave unmasked",transforms=(),layers=False)
prompt_input = gr.Textbox(label="Prompt")
num_inference_steps_input = gr.Slider(minimum=0, maximum=100, value=50, label="Number of Inference Steps")
image_guidance_scale_input = gr.Slider(minimum=0.0, maximum=2.5, value=1.5, label="Image Guidance Scale")
guidance_scale_input = gr.Slider(minimum=0.0, maximum=17.5, value=12.5, label="Guidance Scale")
seed_input = gr.Textbox(value="255", label="Random Seed")

inputs = [image_mask_input, prompt_input, num_inference_steps_input, image_guidance_scale_input, guidance_scale_input,
          seed_input]
outputs = gr.Image(label="Generated Image")


# Custom HTML content
article_html = """
<div style="text-align: center; max-width: 1000px; margin: 20px auto; font-family: Arial, sans-serif;">
  <h2 style="font-weight: 900; font-size: 2.5rem; margin-bottom: 0.5rem;">
    🖼️ UltraEdit for Fine-Grained Image Editing
  </h2>
  <div style="margin-bottom: 1rem;">
    <h3 style="font-weight: 500; font-size: 1.25rem; margin: 0;">
    </h3>
    <p style="font-weight: 400; font-size: 1rem; margin: 0.5rem 0;">
      Haozhe Zhao<sup>1*</sup>, Xiaojian Ma<sup>2*</sup>, Liang Chen<sup>1</sup>, Shuzheng Si<sup>1</sup>, Rujie Wu<sup>1</sup>,
      Kaikai An<sup>1</sup>, Peiyu Yu<sup>3</sup>, Minjia Zhang<sup>4</sup>, Qing Li<sup>2</sup>, Baobao Chang<sup>2</sup>
    </p>
    <p style="font-weight: 400; font-size: 1rem; margin: 0;">
      <sup>1</sup>Peking University, <sup>2</sup>BIGAI, <sup>3</sup>UCLA, <sup>4</sup>UIUC
    </p>
  </div>
  <div style="margin: 1rem 0; display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap;">
    <a href="https://huggingface.co/datasets/BleachNick/UltraEdit" style="display: flex; align-items: center; text-decoration: none; color: blue; font-weight: bold; gap: 0.5rem;">
      <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Dataset_4M" style="height: 20px; vertical-align: middle;"> Dataset
    </a>
    <a href="https://huggingface.co/datasets/BleachNick/UltraEdit_500k" style="display: flex; align-items: center; text-decoration: none; color: blue; font-weight: bold; gap: 0.5rem;">
      <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Dataset_500k" style="height: 20px; vertical-align: middle;"> Dataset_500k
    </a>
     <a href="https://ultra-editing.github.io/" style="display: flex; align-items: center; text-decoration: none; color: blue; font-weight: bold; gap: 0.5rem;">
      <span style="font-size: 20px; vertical-align: middle;">🔗</span> Page
    </a>
    <a href="https://huggingface.co/BleachNick/SD3_UltraEdit_w_mask" style="display: flex; align-items: center; text-decoration: none; color: blue; font-weight: bold; gap: 0.5rem;">
      <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Checkpoint" style="height: 20px; vertical-align: middle;"> Checkpoint
    </a>
    <a href="https://github.com/HaozheZhao/UltraEdit" style="display: flex; align-items: center; text-decoration: none; color: blue; font-weight: bold; gap: 0.5rem;">
      <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 20px; vertical-align: middle;"> GitHub
    </a>
  </div>
  <div style="text-align: left; margin: 0 auto; font-size: 1rem; line-height: 1.5;">
    <p>
    <b>UltraEdit</b> is a dataset designed for fine-grained, instruction-based image editing. It contains over 4 million free-form image editing samples and more than 100,000 region-based image editing samples, automatically generated with real images as anchors. 
    </p>
    <p>
      This demo allows you to perform image editing using the <a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers" style="color: blue; text-decoration: none;">Stable Diffusion 3</a> model trained with this extensive dataset. It supports both free-form (without mask) and region-based (with mask) image editing. Use the sliders to adjust the inference steps and guidance scales, and provide a seed for reproducibility. The image guidance scale of 1.5 and  text guidance scale of 7.5 / 12.5 is a good start for free-from/region-based image editing.
    </p>
  </div>
</div>
"""
html='''
  <div style="text-align: left; margin-top: 2rem; font-size: 0.85rem; color: gray;">
    <p>
      <b>Usage Instructions:</b> You need to upload the images and prompts for editing. Use the pen tool to mark the areas you want to edit. If no region is marked, it will resort to free-form editing.
    </p>
  </div>
'''
demo = gr.Interface(
    fn=generate,
    inputs=inputs,
    outputs=outputs,
    description=article_html,  # Add article parameter
    article = html,
    examples=mask_ex_list
)

demo.queue().launch()

