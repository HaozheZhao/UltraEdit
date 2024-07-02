#!/bin/bash
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
pretrained_model_name_or_path=resolution_512_model_epoch_5_sd3_5e5
ori_model_name_or_path=resolution_512_model_epoch_5_sd3_5e5
resolution=512
train_data_jsonl=mixed_mask_free_1M_6_29.jsonl # mix the redion-basd and free-form image editing data
train_batch_size=8
gradient_accumulation_steps=4
checkpointing_steps=400
validation_step=400

epoch=2
lr=1e-05
output_dir="resolution_${resolution}_model_epoch_${epoch}_conitnue_1e5_with_mask"
accelerate launch --mixed_precision="fp16" --multi_gpu --main_process_port=29555 training/train_sd3_pix2pix.py \
 --pretrained_model_name_or_path ${pretrained_model_name_or_path} \
 --output_dir ${output_dir} \
 --train_data_jsonl ${train_data_jsonl} \
 --resolution ${resolution} \
 --do_mask \
 --num_train_epochs ${epoch} \
 --random_flip \
 --train_batch_size ${train_batch_size} \
 --gradient_accumulation_steps ${gradient_accumulation_steps} \
 --checkpointing_steps ${checkpointing_steps}  \
 --checkpoints_total_limit 2 \
 --learning_rate ${lr} \
 --lr_warmup_steps 500 \
 --conditioning_dropout_prob 0.05 \
 --mixed_precision fp16 \
 --seed 3345 \
 --num_validation_images 4 \
 --validation_step ${validation_step} \
 --dataloader_num_workers 64 \
 --lr_scheduler cosine \
 --report_to wandb \
 --val_image_url input.png \
 --val_mask_url mask_img.png \
 --validation_prompt "What if the horse wears a hat?" \
 --ori_model_name_or_path ${ori_model_name_or_path} \
 --max_sequence_length 256 \




