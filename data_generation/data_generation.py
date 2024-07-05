
import base64

import io
import multiprocessing
import os
import random
import traceback
from argparse import ArgumentParser
from PIL import Image, ImageOps
from multiprocessing import Process
from typing import Tuple, List
import numpy as np
import requests
import time
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label, find_objects, grey_dilation
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from metrics.clip_similarity import ClipSimilarity
from sdxl_p2p_pipeline import Prompt2PromptPipeline,Prompt2PromptImg2ImgPipeline,Prompt2PromptInpaintPipeline
from PIL import Image, ImageDraw, ImageFont
import argparse
import copy

from torchvision.ops import box_convert
import os, sys
import pandas as pd
Image.MAX_IMAGE_PIXELS = 1000000000
from PIL import Image
from einops import rearrange, repeat
import numpy as np


import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
# set to ignore the warning
import warnings
warnings.filterwarnings("ignore")
# get INPAINTING for os.environ
do_inpainting = os.environ.get('INPAINTING', 'False')
# convert to boolean
do_inpainting = do_inpainting.lower() in ['true', '1', 't', 'y', 'yes']
if do_inpainting:
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import box_ops
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util.inference import annotate, predict

def generate_Contours_mask(pil_img):
    # Convert PIL image to OpenCV image (RGB to BGR)
    img_np = np.array(pil_img)
    
    if img_np.max() <= 1:
        img_np = (img_np * 255).astype('uint8')
    
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 60, 60, 7, 15)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get binary image
    _, binary_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw the contours
    mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.uint8)

    # Fill each contour
    for contour in contours:
        # Calculate convex hull 
        hull = cv2.convexHull(contour)
        
        # Draw it (filled) on the mask image
        cv2.drawContours(mask, [hull], -1, (255), -1)
    
    # Convert mask to PIL image
    mask_pil = Image.fromarray(mask)

    return mask_pil

def find_contours_number(mask_image: np.array):
    contours, _ = cv2.findContours(mask_image.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def check_mask_size(mask, min_size=0.01, max_size=0.9):
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_pixels = np.count_nonzero(mask)
    percent = (mask_pixels / total_pixels)
    if min_size < percent < max_size:
        return True
    else:
        return False

def create_mask_image(image_source, detected_boxes):
    # Get the height and width of the source image
    h, w, _ = image_source.shape

    # Create meshgrid
    x = torch.arange(w).reshape(1, -1).expand(h, -1)
    y = torch.arange(h).reshape(-1, 1).expand(-1, w)

    # Convert the detected_boxes to the actual pixel values
    boxes = (detected_boxes * torch.Tensor([w, h, w, h])).type(torch.int)

    # Convert to 'xyxy' format
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # Create a mask for each box without for-loop.
    masks = [((y >= box[1]) & (y <= box[3]) & (x >= box[0]) & (x <= box[2])).numpy() for box in xyxy]

    # Aggregate all masks
    mask = np.logical_or.reduce(masks, axis=0)

    # Convert numpy array to PIL image
    try:
        mask_img = Image.fromarray(mask)
    except:
        mask_img = None
        print("CAN NOT SAM")

    return mask_img
def to_pil(image: torch.Tensor) -> Image.Image:
    image = 255.0 * rearrange(image.cpu().numpy(), "c h w -> h w c")
    image = Image.fromarray(image.astype(np.uint8))
    return image

def gd_load_image(image_path) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if isinstance(image_path, str):
        image_source = Image.open(image_path).convert("RGB")
    elif isinstance(image_path, Image.Image):
        image_source = image_path
    elif image_path is None:
        return None, None
    else:
        raise ValueError("image_path must be a string or a PIL Image")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed
def load_image(image_path, target_size=512):
    # Load the image.
    # try:
    img = Image.open(image_path).convert('RGB')
    
    # Resize while maintaining aspect ratio.
    width, height = img.size
    if width > height:
        ratio = target_size / height
        new_width = int(ratio * width)
        img = img.resize((new_width, target_size))
    else:
        ratio = target_size / width
        new_height = int(ratio * height)
        img = img.resize((target_size, new_height))
    
    # Center-crop the image to 512x512.
    width, height = img.size
    left = (width - target_size) / 2
    top = (height - target_size) / 2
    right = (width + target_size) / 2
    bottom = (height + target_size) / 2
    img = img.crop((left, top, right, bottom))
    # except Exception as e:
    #     print(e)
    #     print(f'Image {image_path} cannot be opened')
    #     # image = Image.new('RGB', (image_size, image_size), (255, 255, 255))
    #     image = Image.open('1.jpg').convert('RGB') # temporal solution
    #     image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    #     image = ImageOps.fit(image, (target_size, target_size), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return img
def compare_prompts(prompt1, prompt2):
    words1 = prompt1.split(' ')
    words2 = prompt2.split(' ')
    difference = [str2 for str1, str2 in zip(words1, words2) if str1 != str2]
    return difference

def generate_images(args,pipe, prompts, init_image, cross_attention_kwargs,cond_mask_img=[],cond_whole_mask_img=[],cond_countor_mask_img=[],countor_number_list=[]):


    try:
        if args.img2img:
            # random choice num_inference_steps in 2 and 4
            num_inference_steps = random.choice([3, 5, 8])
            float_guidance_scale = random.uniform(0.4, 0.8)
            image = pipe(prompts, image=init_image, num_inference_steps=num_inference_steps, strength=0.6, guidance_scale=float_guidance_scale,cross_attention_kwargs=cross_attention_kwargs, output_type = 'pt' ).images
        elif args.do_inpainting  and len(cond_mask_img) == len(init_image) and len(cond_mask_img) == len(prompts) :
            # step to random from 6 to 10
            num_inference_steps = random.choice([10,14])
            float_guidance_scale = random.choice([0.0, 0.2,0.4,0.6])
            soft_mask = random.choice([0.0, 0.1,0.3,0.5,0.7,0.8])
            # soft_mask = 0.1
            # soft_mask = None
            # use cond_mask_img or cond_whole_mask_img
            # mask_list =random.choice([cond_mask_img,cond_whole_mask_img])
            # get True in 50% randomly
            if len(countor_number_list) > 0:
                mask_image = [ ]
                temp_image = [ ]
                for idx, each in enumerate(countor_number_list):
                    if each< 150 and each > 0 :
                        mask_image.append(cond_mask_img[idx])
                        temp_image.append(cond_countor_mask_img[idx])
                    elif each >=150 and each < 500:
                        mask_image.append(cond_countor_mask_img[idx])
                        temp_image.append(cond_whole_mask_img[idx])
                    else:
                        mask_image.append(cond_mask_img[idx])
                        temp_image.append(cond_mask_img[idx])

                using_soft_mask = random.choice([True, False])
                if using_soft_mask:
                    image = pipe(prompts, image=init_image, mask_image=mask_image, num_inference_steps=num_inference_steps, guidance_scale=float_guidance_scale, temp_mask = temp_image ,cross_attention_kwargs=cross_attention_kwargs,soft_mask=soft_mask,output_type = 'pt' ).images
                else:
                    image = pipe(prompts, image=init_image, mask_image=mask_image, num_inference_steps=num_inference_steps, guidance_scale=float_guidance_scale ,cross_attention_kwargs=cross_attention_kwargs,output_type = 'pt' ).images


            
            else:
                use_contour = random.choice([True, False])
                if use_contour:
                    image = pipe(prompts, image=init_image, mask_image=cond_countor_mask_img, num_inference_steps=num_inference_steps, guidance_scale=float_guidance_scale ,cross_attention_kwargs=cross_attention_kwargs,output_type = 'pt' ).images
                else:
                    temp_mask_type = random.choice([True, False])
                    if temp_mask_type:
                        temp_mask = cond_whole_mask_img
                    else: 
                        temp_mask = cond_countor_mask_img

                    image = pipe(prompts, image=init_image, mask_image=cond_mask_img, num_inference_steps=num_inference_steps, guidance_scale=float_guidance_scale, temp_mask = temp_mask ,cross_attention_kwargs=cross_attention_kwargs,soft_mask=soft_mask,output_type = 'pt' ).images


        else:
            # step to random from 1 to 4
            num_inference_steps = random.randint(2, 4)
            float_guidance_scale = random.uniform(0.4, 1.0)
            image = pipe(prompts,  num_inference_steps=num_inference_steps, guidance_scale=float_guidance_scale,cross_attention_kwargs=cross_attention_kwargs, output_type = 'pt' ).images
    except Exception as e:
        traceback.print_exc()
        print("cross_attention_kwargs:", cross_attention_kwargs)
        print("prompt:", prompts)
        return None
    return image
def generate_contour_mask(mask):
    # Convert the mask to grayscale and then to uint8 format
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask_uint8 = mask_gray.astype('uint8')

    # Find contours using OpenCV
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to fill the contours
    contour_mask = np.zeros_like(mask_uint8)

    # Fill the contours on the mask
    for contour in contours:
        cv2.drawContours(contour_mask, [contour], 0, (255), thickness=cv2.FILLED)

    return contour_mask                          
# detect object using grounding DINO
def detect(image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model, 
      image=image, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )
  return boxes 

def segment(image, sam_model, boxes,device):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()
  
@torch.no_grad()
def save_tsv(args, shard_id, shard, device, global_data):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)
    image_size = args.image_size
    # background_images = load_json(args.background_images)
    # CLip score
    clip_similarity = ClipSimilarity(args.clip_model).cuda(device)
    # SDXL_ P2P Pipeline:
    groundingdino_model = None
    sam_predictor = None
    if args.do_inpainting:
        print('Using Prompt2PromptInpaintingPipeline')
        pipe = Prompt2PromptInpaintPipeline.from_pretrained(args.pipeline_ckpt,torch_dtype=torch.float16, variant="fp16").to(device)
        groundingdino_model = load_groundingdino_model(args.groundingdino_config_file, args.groundingdino_checkpoint, device)
        sam_predictor = SamPredictor(build_sam(checkpoint=args.sam_checkpoint).to(device))
    elif args.img2img:
            print('Using Prompt2PromptImg2ImgPipeline')
            pipe = Prompt2PromptImg2ImgPipeline.from_pretrained(args.pipeline_ckpt,torch_dtype=torch.float16, variant="fp16").to(device)
    else:
        print('Using Prompt2PromptPipeline')
        pipe = Prompt2PromptPipeline.from_pretrained(args.pipeline_ckpt,torch_dtype=torch.float16, variant="fp16").to(device)
    pipe.unet.config.addition_embed_type =None
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.safety_checker = lambda images, clip_input: (images, False)
    except Exception as e:
        print("Error: ",e)
        pass
    # pipe.eval()


    cnt = 0
    image_paths = []
    prompt = []
    new_prompt = []
    edit= []
    data_path= []
    print("=====================Strat training========================")
    for s_data in tqdm(shard):
        if args.do_inpainting:
            init_image,global_idx,gd_image,gd_image_transformed= s_data
        else:
            init_image,global_idx = s_data
        if init_image is None and global_idx is None:
            continue
        cond_img=[]
        cond_mask_img=[]
        cond_whole_mask_img=[]
        cond_countor_mask_img=[]
        countor_number_list=[]
        replace_countor_number_list = []
        prompts=[]
        edit_instruction = []
        replace_prompt=[]
        replace_text_prompt =[]
        cond_text_prompt = []
        replace_cond_img = [] # batch generate do not support the replace edit
        replace_cond_mask_img = [] # batch generate do not support the replace edit
        replace_cond_whole_mask_img = [] # batch generate do not support the replace edit
        replace_cond_countor_mask_img = [] # batch generate do not support the replace edit
        replace_edit_instruction =[]
        for num, (img,idx) in enumerate(zip(init_image,global_idx)):
            if args.img2img or args.do_inpainting:
                if img is None:
                    continue
            data_temp = global_data[idx]
            input_text_length = len(data_temp['input_text'].split(' '))
            output_text_length = len(data_temp['output_text'].split(' '))
            if input_text_length<=60 and output_text_length <= 60:
                if args.do_replace:
                    if input_text_length == output_text_length:
                        replace_prompt.append([data_temp['input_text'], data_temp['output_text']])
                        if args.img2img:
                            replace_cond_img.append(img)
                        elif args.do_inpainting:    
                            # (image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25)
                            text_prompt = data_temp['edit_object']
                            # if text_prompt[-1] != '.':
                            #     text_prompt += '.'
                            try:
                                if'NONE' in text_prompt:
                                    # make mask_img as the mask of whole image in size of args.image_size
                                    mask_img = Image.new('L', (args.image_size, args.image_size), color=255)
                                    mask_contour = mask_img
                                    mask_box = mask_img
                                    contour_number = 0
                                else:
                                    detect_boxes = detect(gd_image_transformed[num], text_prompt, groundingdino_model,box_threshold = args.gd_box_threshold, text_threshold = args.gd_text_threshold)
                                    mask_box = create_mask_image(gd_image[num], detect_boxes)
                                    if mask_box is None:
                                        prompts = prompts[:-2]
                                        edit_instruction = edit_instruction[:-1]
                                        continue
                                    segmented_frame_masks = segment(gd_image[num], sam_predictor, boxes=detect_boxes,device=device)
                                    merged_mask = torch.max(segmented_frame_masks, 0)[0][0]
                                    mask = merged_mask.cpu().numpy()
                                    # inverted_mask = ((1 - mask) * 255).astype(np.uint8)
                                    # generate_contour_mask(mask)
                                    mask_img = Image.fromarray(mask)
                                    if not check_mask_size(mask, min_size=0.01, max_size=0.9):
                                        replace_prompt = replace_prompt[:-2]
                                        continue
                                    contour_number = find_contours_number(mask)
                                    if contour_number > 500:
                                        replace_prompt = replace_prompt[:-2]
                                        continue
                                    mask_contour = generate_Contours_mask(mask_img)
                            except Exception as e:
                                # wipe out the append prompts and edit_instruction
                                replace_prompt = replace_prompt[:-2]
                                print(e)
                                traceback.print_exc()
                                continue
                            
                            replace_cond_mask_img.append(mask_img)
                            replace_cond_whole_mask_img.append(mask_box)
                            replace_cond_countor_mask_img.append(mask_contour)
                            replace_cond_img.append(img)
                            replace_text_prompt.append(text_prompt)
                            replace_countor_number_list.append(contour_number)
                            replace_countor_number_list.append(contour_number)
                        else:
                            replace_cond_img.append(None)
                        replace_edit_instruction.append(data_temp['edit_instruction'])

                    else:
                        prompts.extend([data_temp['input_text'], data_temp['output_text']])
                        edit_instruction.append(data_temp['edit_instruction'])
                        if args.img2img:
                            cond_img.append(img)
                            cond_img.append(img)
                        elif args.do_inpainting:
                            text_prompt = data_temp['edit_object']
                            # if text_prompt[-1] != '.':
                            #     text_prompt += '.'
                            try:
                                if'NONE' in text_prompt:
                                    # make mask_img as the mask of whole image in size of args.image_size
                                    mask_img = Image.new('L', (args.image_size, args.image_size), color=255)
                                    mask_contour = mask_img
                                    mask_box = mask_img
                                    contour_number = 0
                                else:
                                    detect_boxes = detect(gd_image_transformed[num], text_prompt, groundingdino_model,box_threshold = args.gd_box_threshold, text_threshold = args.gd_text_threshold)
                                    mask_box = create_mask_image(gd_image[num], detect_boxes)
                                    if mask_box is None:
                                        prompts = prompts[:-2]
                                        edit_instruction = edit_instruction[:-1]
                                        continue
                                    segmented_frame_masks = segment(gd_image[num], sam_predictor, boxes=detect_boxes,device=device)
                                    merged_mask = torch.max(segmented_frame_masks, 0)[0][0]
                                    mask = merged_mask.cpu().numpy()
                                    # inverted_mask = ((1 - mask) * 255).astype(np.uint8)
                                    mask_img = Image.fromarray(mask)
                                    if not check_mask_size(mask, min_size=0.01, max_size=0.9):
                                        prompts = prompts[:-2]
                                        edit_instruction = edit_instruction[:-1]
                                        continue
                                    contour_number = find_contours_number(mask)
                                    if contour_number > 500:
                                        prompts = prompts[:-2]
                                        edit_instruction = edit_instruction[:-1]
                                        continue
                                    mask_contour = generate_Contours_mask(mask_img)
                            except Exception as e:
                                # wipe out the append prompts and edit_instruction
                                prompts = prompts[:-2]
                                edit_instruction = edit_instruction[:-1]
                                print(e)
                                traceback.print_exc()
                                continue
                            cond_mask_img.append(mask_img)
                            cond_mask_img.append(mask_img)
                            cond_whole_mask_img.append(mask_box)
                            cond_whole_mask_img.append(mask_box)
                            cond_countor_mask_img.append(mask_contour)
                            cond_countor_mask_img.append(mask_contour)
                            cond_img.append(img)
                            cond_img.append(img)
                            cond_text_prompt.append(text_prompt)
                            countor_number_list.append(contour_number)
                            countor_number_list.append(contour_number)
                        
                else:
                    prompts.extend([data_temp['input_text'], data_temp['output_text']])
                    edit_instruction.append(data_temp['edit_instruction'])
                    if args.img2img:
                        cond_img.append(img)
                        cond_img.append(img)
                    elif args.do_inpainting:
                        text_prompt = data_temp['edit_object']
                        # if text_prompt[-1] != '.':
                        #     text_prompt += '.'
                        try:
                            if'NONE' in text_prompt:
                                # make mask_img as the mask of whole image in size of args.image_size
                                mask_img = Image.new('L', (args.image_size, args.image_size), color=255)
                                mask_contour = mask_img
                                mask_box = mask_img
                                contour_number = 0
                            else:
                                detect_boxes = detect(gd_image_transformed[num], text_prompt, groundingdino_model,box_threshold = args.gd_box_threshold, text_threshold = args.gd_text_threshold)
                                mask_box = create_mask_image(gd_image[num], detect_boxes)
                                if mask_box is None:
                                    prompts = prompts[:-2]
                                    edit_instruction = edit_instruction[:-1]
                                    continue
                                segmented_frame_masks = segment(gd_image[num], sam_predictor, boxes=detect_boxes,device=device)
                                merged_mask = torch.max(segmented_frame_masks, 0)[0][0]
                                mask = merged_mask.cpu().numpy()
                                # inverted_mask = ((1 - mask) * 255).astype(np.uint8)
                                mask_img = Image.fromarray(mask)
                                if not check_mask_size(mask, min_size=0.01, max_size=0.9):
                                    prompts = prompts[:-2]
                                    edit_instruction = edit_instruction[:-1]                     
                                    continue
                                contour_number = find_contours_number(mask)
                                if contour_number > 500 or contour_number< 4:
                                    prompts = prompts[:-2]
                                    edit_instruction = edit_instruction[:-1]
                                    continue
                                mask_contour = generate_Contours_mask(mask_img)
                        except Exception as e:
                            # wipe out the append prompts and edit_instruction
                            prompts = prompts[:-2]
                            edit_instruction = edit_instruction[:-1]
                            print(e)
                            traceback.print_exc()
                            continue

                        cond_mask_img.append(mask_img)
                        cond_mask_img.append(mask_img)
                        cond_whole_mask_img.append(mask_box)
                        cond_whole_mask_img.append(mask_box)
                        cond_countor_mask_img.append(mask_contour)
                        cond_countor_mask_img.append(mask_contour)
                        cond_img.append(img)
                        cond_img.append(img)
                        cond_text_prompt.append(text_prompt)
                        countor_number_list.append(contour_number)
                        countor_number_list.append(contour_number)
            else:
                print("length input text: ", input_text_length)
                print("length output text: ", output_text_length)

        if cnt % 1000 == 0:
            # close previous file if any
            # if cnt > 0:
            #     f.close()
            # f = open(os.path.join(args.output_dir, f"cnt_{args.data_split_start}_{args.data_split_end}_{args.machine_id}_{shard_id}_{cnt // 1000}.tsv"), "w", encoding='utf-8')
            folder = os.path.join(args.output_dir, f"cnt_{args.data_split_start}_{args.data_split_end}_{args.machine_id}_{shard_id}_{cnt // 1000}")
            if not os.path.exists(folder):
                os.makedirs(folder,exist_ok=True)

            if cnt > 0:
                pd.DataFrame(
                    {
                        # 'image_paths': image_paths,
                        'data_path': data_path,    
                        'prompt': prompt,
                        'new_prompt': new_prompt,
                        'edit': edit
                    }
                ).to_csv(os.path.join(args.output_dir, f"cnt_{args.data_split_start}_{args.data_split_end}_{args.machine_id}_{shard_id}_{cnt // 1000}.csv"),  index=False)
                # image_paths = []
                data_path= []
                prompt = []
                new_prompt = []
                edit = []
            # f = open(os.path.join(args.output_dir, f"cnt_{args.machine_id}_{shard_id}_{cnt // 1000}.tsv"), "w", encoding='utf-8')
        
        
        # cnt += 1
        # results = [{}]*args.batch_size
        new_results ={
            i:{}
            for i in range(args.batch_size)
        }
        # throw error if the length of edit_instruction + replace_edit_instruction != batch_size
        # assert len(edit_instruction) + len(replace_edit_instruction) == args.batch_size
        start_count = 0
        if len(replace_edit_instruction) > 0:
            start_count = len(replace_edit_instruction)
        # cnt+= args.batch_size
        while len(new_results[0]) < args.n_samples:
            seed = torch.randint(1 << 32, ()).item()
            if seed in new_results:
                continue
            # set random seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)


            p2p_threshold = args.min_p2p + torch.rand(()).item() * (args.max_p2p - args.min_p2p)
            cfg_scale = args.min_cfg + torch.rand(()).item() * (args.max_cfg - args.min_cfg)

        # image = pipe(prompts, num_inference_steps=2,   guidance_scale=0.0,cross_attention_kwargs=cross_attention_kwargs,)

            # do replace one by one:
            if len(replace_prompt)>0 and args.do_replace:
                # for replace_p, replace_img, replace_edit in zip(replace_prompt, replace_cond_img, replace_edit_instruction):# make it enumerate

                for idx, (replace_p, replace_img, replace_edit) in enumerate(zip(replace_prompt, replace_cond_img, replace_edit_instruction)):
                    cross_replace = compare_prompts(replace_p[0],replace_p[1])
                    if len(cross_replace)<=5:
                        cross_replace = ' '.join(cross_replace)
                        n_cross_replace = {"default_": 1.0, str(cross_replace): p2p_threshold}
                    else:
                        n_cross_replace = p2p_threshold
                    cross_attention_kwargs = {
                                "edit_type": "replace",
                                "n_self_replace": p2p_threshold,
                                "n_cross_replace": n_cross_replace ,
                                }
                    if args.do_inpainting and len(replace_cond_mask_img) == 0 :
                        continue
                    image = generate_images(args,pipe, replace_p, replace_img, cross_attention_kwargs,replace_cond_mask_img,cond_whole_mask_img,replace_cond_countor_mask_img,replace_countor_number_list)
                    if image is None:
                        print("replace_p",replace_p)
                        continue
                    original_image,edit_image = image[0],image[1]
                    clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image,dinov2_sim,ssim = clip_similarity(
                        original_image.unsqueeze(0), edit_image.unsqueeze(0), [replace_p[0]], [replace_p[1]]
                            )
                    new_results[idx][seed] = dict(
                        image_0=to_pil(original_image),
                        image_1=to_pil(edit_image),
                        p2p_threshold=p2p_threshold,
                        clip_sim_0=clip_sim_0[0].item(),
                        clip_sim_1=clip_sim_1[0].item(),
                        clip_sim_dir=clip_sim_dir[0].item(),
                        clip_sim_image=clip_sim_image[0].item(),
                        dinov2_sim = dinov2_sim[0].item(),
                        ssim = ssim,
                        caption=replace_p[0],
                        new_caption=replace_p[1],
                        edit_instance= replace_edit
                    )
                    if args.do_inpainting:
                        new_results[num_idx+start_count][seed]['img_mask'] = replace_cond_mask_img[0]
                        new_results[num_idx+start_count][seed]['edit_object'] = replace_text_prompt[idx]
            # batch refine:
            
            cross_attention_kwargs = {
            "edit_type": "refine",
            "n_self_replace": p2p_threshold,
            "n_cross_replace": p2p_threshold
            }
            if len(prompts)>0:
                if args.do_inpainting and len(cond_mask_img) == 0 :
                        break
                # if args.img2img:
                #     if isinstance(cond_img, list):
                #         if cond_img[0].shape[0] == 3:
                #             cond_img = torch.stack(cond_img)
                #         else:
                #             cond_img = torch.cat(cond_img)
                image = generate_images(args,pipe, prompts, cond_img, cross_attention_kwargs,cond_mask_img,cond_whole_mask_img,cond_countor_mask_img,countor_number_list)
                # spilt image into pairs, the first is original image, the second is edited image
                if image is None:
                    break
            else:
                break
            # make it enumerate
            # for i in range(0,len(image),2):
            for num_idx, i in enumerate(range(0,len(image),2)):
                original_image,edit_image = image[i],image[i+1]
                clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image,dinov2_sim,ssim = clip_similarity(
                    original_image.unsqueeze(0), edit_image.unsqueeze(0), [prompts[i]], [prompts[i+1]]
                        )
                new_results[num_idx+start_count][seed] = dict(
                    image_0=to_pil(original_image),
                    image_1=to_pil(edit_image),
                    p2p_threshold=p2p_threshold,
                    clip_sim_0=clip_sim_0[0].item(),
                    clip_sim_1=clip_sim_1[0].item(),
                    clip_sim_dir=clip_sim_dir[0].item(),
                    clip_sim_image=clip_sim_image[0].item(),
                    dinov2_sim = dinov2_sim[0].item(),
                    ssim = ssim,
                    caption=prompts[i],
                    new_caption=prompts[i+1],
                    edit_instance= edit_instruction[i//2],
                )
                if args.do_inpainting:
                    new_results[num_idx+start_count][seed]['img_mask'] = cond_mask_img[i]
                    new_results[num_idx+start_count][seed]['edit_object'] = cond_text_prompt[i//2]


        for re_idx,(k,result) in enumerate(new_results.items()):
            cnt+=1
            metadata = [
                    (result_sample["clip_sim_dir"], seed)
                    for seed, result_sample in result.items()
                    if result_sample["clip_sim_image"] >= args.clip_img_threshold
                    and result_sample["clip_sim_dir"] >= args.clip_dir_threshold
                    and result_sample["clip_sim_0"] >= args.clip_threshold
                    and result_sample["clip_sim_1"] >= args.clip_threshold
                    and result_sample["dinov2_sim"] >= args.dinov2_sim_threshold
                    # and (result_sample["dinov2_sim"] <= args.max_dinov2_sim_threshold or result_sample["clip_sim_dir"]>=args.clip_dir_threshold+0.05 )
                    # and (result_sample["ssim"] <= args.max_ssim_threshold or result_sample["clip_sim_dir"]>=args.clip_dir_threshold+0.1 )

                ] 
            metadata.sort(reverse=True)

            if len(metadata) > 0:
                samples = metadata[: args.max_out_samples]
                if args.do_inpainting:
                    if len(samples) < args.max_out_samples:
                        continue
                prompt_dir = os.path.join(folder, str(cnt).zfill(7))
                prompt.append(result[metadata[0][1]]['caption'])
                new_prompt.append(result[metadata[0][1]]['new_caption'])
                edit.append(result[metadata[0][1]]['edit_instance'])
                # image_paths.append(metadata[0]['input_image'])
                data_path.append(prompt_dir)
                os.makedirs(prompt_dir, exist_ok=True)

                for _, seed in samples:
                    result_sample = result[seed]
                    image_0 = result_sample.pop("image_0")
                    image_1 = result_sample.pop("image_1")
                    if args.do_inpainting:
                        img_mask = result_sample.pop("img_mask")
                    try:
                        image_0.save(os.path.join(prompt_dir, f"{seed}_0.jpg"), quality=100)
                        image_1.save(os.path.join(prompt_dir, f"{seed}_1.jpg"), quality=100)
                        if args.do_inpainting:
                            img_mask.save(os.path.join(prompt_dir, f"{seed}_mask.jpg"), quality=100)
                    except Exception as e:
                        print(e)
                        print(f'Image cannot be saved, sleep 5s')
                        time.sleep(5)
                        try:    
                            image_0.save(os.path.join(prompt_dir, f"{seed}_0.jpg"), quality=100)
                            image_1.save(os.path.join(prompt_dir, f"{seed}_1.jpg"), quality=100)
                            if args.do_inpainting:
                                img_mask.save(os.path.join(prompt_dir, f"{seed}_mask.jpg"), quality=100)
                        except Exception as e:
                            print(e)
                            print('Image cannot be saved again, skipping...')
                            continue 
                    with open(os.path.join(prompt_dir, f"metadata.jsonl"), "a") as fp:
                        fp.write(f"{json.dumps(dict(seed=seed, **result_sample))}\n")

# class OpenImageDataset(Dataset):
#     def __init__(self, url_data):
#         self.data = url_data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         try:
#             # items = self.data[idx].split(',')
#             # image = Image.open(requests.get(items[2], stream=True).raw).convert('RGB')
#             data_temp = self.data[idx]
#             img_path = data_temp['input_image']
#             global_idx = data_temp['global_idx']
#             # try:
#             image = Image.open(img_path).convert('RGB')
#             # except Exception as e:
#             #     print(e)
#             #     print(f'Image {img_path} cannot be opened')
#             #     # image = Image.new('RGB', (224, 224), (255, 255, 255))
#             #     image = Image.open('1.jpg ').convert('RGB') # temporal solution
            

#             # caption
#             width, height = image.size
#             shortest_side = min(width, height)
#             left = (width - shortest_side) // 2
#             top = (height - shortest_side) // 2
#             right = left + shortest_side
#             bottom = top + shortest_side
#             image = image.crop((left, top, right, bottom))
#             return image,global_idx
#         except:
#             return None

import requests
from PIL import Image
from torch.utils.data import Dataset
class NoneImageDataset(Dataset):
    def __init__(self, url_data):
        self.data = url_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_temp = self.data[idx]
        img_path = data_temp['input_image']
        global_idx = data_temp['global_idx']
            
        return img_path, global_idx

class OpenImageDataset(Dataset):
    def __init__(self, url_data):
        self.data = url_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_temp = self.data[idx]
        img_path = data_temp['input_image']
        global_idx = data_temp['global_idx']

        try:
            image = load_image(img_path)
        except Exception as e:
            print(e)
            print(f'Image {img_path} cannot be opened')
            image = None
        
        if image is not None:
            # caption
            width, height = image.size
            shortest_side = min(width, height)
            left = (width - shortest_side) // 2
            top = (height - shortest_side) // 2
            right = left + shortest_side
            bottom = top + shortest_side
            image = image.crop((left, top, right, bottom))
            
        return image, global_idx

class RegionDataset(Dataset):
    def __init__(self, url_data):
        self.data = url_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_temp = self.data[idx]
        img_path = data_temp['input_image']
        global_idx = data_temp['global_idx']

        try:
            image = load_image(img_path)
        except Exception as e:
            print(e)
            print(f'Image {img_path} cannot be opened')
            image = None
        
        if image is not None:
            # caption
            width, height = image.size
            shortest_side = min(width, height)
            left = (width - shortest_side) // 2
            top = (height - shortest_side) // 2
            right = left + shortest_side
            bottom = top + shortest_side
            image = image.crop((left, top, right, bottom))
        gd_image_source, image_transformed = gd_load_image(image)
        
        return image, global_idx,gd_image_source, image_transformed


# def load_images(image_path, image_size=384):
#     try:
#         image = Image.open(image_path).convert('RGB')
#         image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
#         image = ImageOps.fit(image, (image_size, image_size), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
#     except Exception as e:
#         print(e)
#         print(f'Image {image_path} cannot be opened')
#         # image = Image.new('RGB', (image_size, image_size), (255, 255, 255))
#         image = Image.open('1.jpg').convert('RGB') # temporal solution
#         image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
#         image = ImageOps.fit(image, (image_size, image_size), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        
#     return np.array(image)

# def collate_fn(batch):
#     return batch[0] if batch is not None else None

def collate_fn(batch):
    images = []
    labels = []
    gd_image_source = []
    image_transformed = []
    if batch is None:
        return None,None
    # elif  batch is not None and len(batch) ==1:
    #     return batch[0]
    else:
        for item in batch:
            if item is None:
                continue
            if len(item) == 2:
                img, lbl = item
                images.append(img)
                labels.append(lbl)
            else:
                img, lbl, gd_img, gd_img_tran = item
                images.append(img)
                labels.append(lbl)
                gd_image_source.append(gd_img)
                image_transformed.append(gd_img_tran)
        if len(gd_image_source) == 0:
            return images, labels
        else:
            return images, labels,gd_image_source,image_transformed
    
# def collate_fn(batch):
    # # Handle the condition where getitem returns None
    # batch = [item for item in batch if item is not None]
    
    # # separate the images and the labels 
    # images, labels = zip(*batch)

    # # pad images if they are not of same size and stack all images & labels
    # images = torch.stack(images)
    # labels = torch.stack(labels)

    # return images, labels


import json
import random

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        return json.load(f)
def load_groundingdino_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    model = model.to(device)
    return model
def main():
    """Parse commandline arguments."""
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default='/path/to/image_ids_and_rotation.csv')
    parser.add_argument('--background_images', type=str, default='image.csv')
    parser.add_argument('--img2img',action='store_true')
    parser.add_argument('--do_replace',action='store_true')
    parser.add_argument('--do_inpainting',action='store_true')
    parser.add_argument('--groundingdino_checkpoint',type=str, default='../../model2/grounding_dino/groundingdino_swinb_cogcoor.pthc')
    parser.add_argument('--groundingdino_config_file',type=str, default='Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py')
    parser.add_argument('--sam_checkpoint',type=str, default='../../model2/sam/sam_vit_h_4b8939.pth')
    parser.add_argument('--gd_box_threshold',type=float, default=0.3)
    parser.add_argument('--gd_text_threshold',type=float, default=0.25)


    parser.add_argument('--output-dir', type=str, default='/path/to/output-dir/')
    parser.add_argument('--num-process', type=int, default=8)
    parser.add_argument('--cuda-device',  nargs='+', type=int,default=[0, 1, 2, 3,4,5,6,7] )
    parser.add_argument('--num-machine', type=int, default=1)
    parser.add_argument('--machine-id', type=int, default=0)

    parser.add_argument('--max-seq-len', type=int, default=None)
    parser.add_argument('--max-new-tokens', type=int, default=10)

    parser.add_argument('--pipeline_ckpt', type=str, default="sdxl-turbo")
    parser.add_argument('--clip_model', type=str, default="ViT-L-14.pt")

    
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=0)
    parser.add_argument('--n_samples', type=float, default=10)
    parser.add_argument(
        "--min-p2p",
        type=float,
        default=0.1,
        help="Min prompt2prompt threshold (portion of denoising for which to fix self attention maps).",
    )
    parser.add_argument(
        "--max-p2p",
        type=float,
        default=0.9,
        help="Max prompt2prompt threshold (portion of denoising for which to fix self attention maps).",
    )
    parser.add_argument(
        "--max-out-samples",
        type=int,
        default=4,
        help="Max number of output samples to save per prompt (after CLIP filtering).",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.2,
        help="CLIP threshold for text-image similarity of each image.",
    )
    parser.add_argument(
        "--clip-dir-threshold",
        type=float,
        default=0.2,
        help="Directional CLIP threshold for similarity of change between pairs of text and pairs of images.",
    )
    parser.add_argument(
        "--min-cfg",
        type=float,
        default=7.5,
        help="Min classifier free guidance scale.",
    )
    parser.add_argument(
        "--max-cfg",
        type=float,
        default=15,
        help="Max classifier free guidance scale.",
    )
    parser.add_argument(
        "--clip-img-threshold",
        type=float,
        default=0.8,
        help="CLIP threshold for image-image similarity.",
    )
    parser.add_argument(
        "--dinov2-sim-threshold",
        type=float,
        default=0.3,
        help="DINOv2 threshold for image-image similarity.",
    )
    # max_dinov2_sim_threshold
    parser.add_argument(
        "--max-dinov2-sim-threshold",
        type=float,
        default=0.9,
        help="Max DINOv2 threshold for image-image similarity.",
    )
    parser.add_argument(
        "--max-ssim-threshold",
        type=float,
        default=0.5,
        help="max SSIM threshold for image-image similarity.",
    )
    parser.add_argument('--seed', type=int, default=324)
    parser.add_argument('--do-sample', type=bool, default=True)
    parser.add_argument('--use-cache', type=bool, default=True)
    parser.add_argument('--trust-remote-code', type=bool, default=True)
    parser.add_argument('--attn-impl', type=str, default='torch')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--data_split_start', type=int, default=0)
    parser.add_argument('--data_split_end', type=int, default=1000000000)
    parser.add_argument('--image_size', type=int, default=384)
    args = parser.parse_args()




    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # with open(args.data_dir, 'r', encoding='utf8') as f:
    #     url_data = f.read().strip().split('\n')
    url_data = load_jsonl(args.data_dir)
    import copy
    # global global_data 
    global_data = copy.deepcopy(url_data)
    # random.shuffle(url_data)
    global_data = {
        each['global_idx']: each
        for each in global_data
    }
    url_data = url_data[args.data_split_start:args.data_split_end]
    # split into 8 machine, and pick the part of machine_id
    url_data = url_data[args.machine_id::args.num_machine]
    print(f'Processing {len(url_data)} images')
    # split url data into shards
    url_data = [url_data[i::args.num_process] for i in range(args.num_process)]
    if args.img2img:
        dataloaders = [
            DataLoader(
                OpenImageDataset(url_data[i]),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=12,
                pin_memory=True,
                persistent_workers=True,
                drop_last=False,
                prefetch_factor=4,
                collate_fn=collate_fn
            )
            for i in range(args.num_process)
        ]
    elif args.do_inpainting:
        dataloaders = [
            DataLoader(
                RegionDataset(url_data[i]),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=12,
                pin_memory=True,
                persistent_workers=True,
                drop_last=False,
                prefetch_factor=4,
                collate_fn=collate_fn
            )
            for i in range(args.num_process)
        ]
    else:
        dataloaders = [
            DataLoader(
                NoneImageDataset(url_data[i]),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=12,
                pin_memory=True,
                persistent_workers=True,
                drop_last=False,
                prefetch_factor=4,
                collate_fn=collate_fn
            )
            for i in range(args.num_process)
        ]

    multiprocessing.set_start_method('spawn')
    processes = []
    # cuda_device =  [int(x) for x in args.cuda_device.split(',')]
    cuda_device = args.cuda_device
    
    for shard_id, shard in enumerate(dataloaders):
        p = Process(
            target=save_tsv,
            args=(
                args,
                shard_id,
                shard,
                torch.device('cuda:{}'.format(cuda_device[shard_id % len(cuda_device)])),
                global_data
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('Done!')


if __name__ == '__main__':
    main()
