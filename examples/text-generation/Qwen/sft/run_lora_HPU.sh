#!/bin/bash
MY_SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"

MODEL_NAME_OR_PATH="/data/qwen1.5-7b-chat"
OUTPUT_PATH="${MY_SCRIPT_DIR}/output_7b_lora"
DS_CONFIG="${MY_SCRIPT_DIR}/deepspeed_zero_1.json"

LOGGING_STEPS=1

hostfile=""
deepspeed --num_nodes 1 \
    --num_gpus 8 \
    --no_local_rank \
    --hostfile=$hostfile finetune.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --report_to "none" \
    --data_path "belle_chat_ramdon_10k.json" \
    --output_dir ${OUTPUT_PATH} \
    --model_max_length 512 \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 8 \
    --save_strategy no \
    --evaluation_strategy no \
    --learning_rate 3e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps ${LOGGING_STEPS} \
    --bf16 True \
    --deepspeed ${DS_CONFIG} \
    --gradient_checkpointing True \
    --use_lora True \
    --lora_r 4 \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
    --use_lazy_mode \
    --throughput_warmup_steps 3 \
    --gaudi_config_name "gaudi_config.json" \
    --use_habana \
    --use_flash_attention True \
    --flash_attention_causal_mask True
#    --attn_softmax_bf16 True \

cp ${MODEL_NAME_OR_PATH}/generation_config.json ${OUTPUT_PATH}/
cp ${MODEL_NAME_OR_PATH}/tokenizer_config.json ${OUTPUT_PATH}/

