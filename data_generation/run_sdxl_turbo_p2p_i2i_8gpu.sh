#!/bin/bash
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python data_generation.py \
--data-dir caption_data.jsonl \
--output-dir sdxl_turbo_p2p_img  \
--num-process 16 \
--image_size 512 \
--cuda-device 0 1 2 3 4 5 6 7 \
--n_samples 30 \
--pipeline_ckpt model/sdxl-turbo \
--clip_model model/ViT-L-14.pt \
--batch_size 7 \
--data_split_start 0 \
--data_split_end 100000 \
--clip-img-threshold 0.75 \
--clip-dir-threshold 0.22 \
--dinov2-sim-threshold 0.4 \
--max-p2p 0.9 \
--img2img \



