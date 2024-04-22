LOWER_LIST=ops_bf16.txt python run_generation.py \
--model_name_or_path state-spaces/mamba-130m-hf \
--use_hpu_graphs \
--batch_size 2048 \
--bf16 \
--max_new_tokens 128 \
--max_input_tokens 128 2>&1 | tee log_mamba.txt
