#!/bin/bash
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export INPAINTING=True
groundingdino_checkpoint=model/grounding_dino/groundingdino_swinb_cogcoor.pth
groundingdino_config_file=.Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py
sam_checkpoint=model/sam/sam_vit_h_4b8939.pth
gd_box_threshold=0.3
gd_text_threshold=0.25
python data_generation.py \
--data-dircaption_list_w_edit_oblect.jsonl \
--output-dir mask_based_sdxl_turbo_p2p_img \
--num-process 16 \
--image_size 512 \
--cuda-device 0 1 2 3 4 5 6 7 \
--n_samples 100 \
--pipeline_ckpt model/sdxl-turbo \
--clip_model model/ViT-L-14.pt \
--batch_size 6 \
--data_split_start 0 \
--data_split_end 100000 \
--clip-img-threshold 0.7 \
--clip-dir-threshold 0.22 \
--dinov2-sim-threshold 0.4 \
--max-dinov2-sim-threshold 0.9 \
--max-ssim-threshold 0.9 \
--max-p2p 0.9 \
--do_inpainting \
--groundingdino_checkpoint ${groundingdino_checkpoint} \
--groundingdino_config_file ${groundingdino_config_file} \
--sam_checkpoint ${sam_checkpoint} \
--gd_box_threshold ${gd_box_threshold} \
--gd_text_threshold ${gd_text_threshold} \
--max-out-samples 3 \
--seed 4324 \
