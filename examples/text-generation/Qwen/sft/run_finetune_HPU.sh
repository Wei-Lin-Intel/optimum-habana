#!/bin/bash
MY_SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"

MODEL_NAME_OR_PATH="/data/qwen1.5-14b-chat"
OUTPUT_PATH="${MY_SCRIPT_DIR}/output_14b"
DS_CONFIG="${MY_SCRIPT_DIR}/deepspeed_zero_2.json"

LOGGING_STEPS=1

hostfile=""
deepspeed --num_nodes 1 \
    --num_gpus 8 \
    --no_local_rank \
    --hostfile=$hostfile finetune.py  \
    --report_to "none" \
    --data_path "belle_chat_ramdon_10k.json" \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_PATH} \
    --model_max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --max_grad_norm 1.0 \
    --logging_steps ${LOGGING_STEPS} \
    --gradient_checkpointing True \
    --deepspeed ${DS_CONFIG} \
    --use_lora False \
    --bf16 True \
    --gaudi_config_name "gaudi_config.json" \
    --use_lazy_mode \
    --throughput_warmup_steps 10 \
    --use_habana \
    --use_flash_attention

cp ${MODEL_NAME_OR_PATH}/generation_config.json ${OUTPUT_PATH}/
cp ${MODEL_NAME_OR_PATH}/tokenizer_config.json ${OUTPUT_PATH}/
