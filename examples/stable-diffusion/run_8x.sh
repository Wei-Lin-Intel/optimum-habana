#LOG_LEVEL_ALL=3 HABANA_LOGS=SDXL_8x_mediapipe


PT_HPU_RECIPE_CACHE_CONFIG=/tmp/stdxl_recipe_cache_ss,True,1024  \
python ../gaudi_spawn.py --world_size 8 --use_mpi --master_port 29500 train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path stabilityai/sdxl-vae \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --resolution 512 \
  --crop_resolution 512 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size 16 \
  --max_train_steps 336 \
  --learning_rate 1e-05 \
  --max_grad_norm 1 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --output_dir sdxl-pokemon-model \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3 \
  --dataloader_num_workers 8 \
  --bf16 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --validation_prompt="a robotic cat with wings" \
  --validation_epochs 48  \
  --checkpointing_steps 500 --mediapipe 2>&1 | tee log_8x_r512_nomediapipe_mar7_docker383_numpyspeedup_nanchk1.txt
  
  #log_8x_r512_nomediapipe_mar5_docker383.txt
