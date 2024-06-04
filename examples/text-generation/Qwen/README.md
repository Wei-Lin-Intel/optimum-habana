<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Inference and Training Benchmark for Qwen Models

This guidance is used to perform the inference and finetune (including lora) benchmark on Gaudi2 for Qwen1.5 and the future release with the same model type in HuggingFace [this link](https://https://github.com/huggingface/transformers/tree/v4.38-release/src/transformers/models/qwen2). The SFT script is adopted from the official Qwen1.5 [repo](https://github.com/QwenLM/Qwen1.5/tree/main/examples/sft).


## Requirements

Step1, use the docker image of Synapse 1.15.1 [Reference](https://docs.habana.ai/en/v1.15.1/) because this repo is based on this version. It is recommended to use Ubuntu 22.04 image:
```bash
docker pull vault.habana.ai/gaudi-docker/1.15.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest
``` 

Step2, you should install this optimum-habana:
```bash
pip install git+https://github.com/Wei-Lin-Intel/optimum-habana.git@v1.11-release
```

Step3, you need to install HabanaAI DeepSpeed [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html) as follows:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.15.1
```

Step4, you should install the requirements:
```bash
pip install -r requirements.txt
```


## Inference

In this section, we present how to benchmark the Qwen model (>=1.5) on Habana Gaudi2 with this script. Theoretically the following commands can support all the models with the `qwen2` model dtype.

To run generation on a single Gaudi2, you can launch the script as follows (supposing your model path is /data/qwen1.5-7b):
```bash
cd optimum-habana/examples/text-generation

python3 run_generation.py \
--model_name_or_path /data/qwen1.5-7b \
--max_input_tokens 2048 \
--max_new_tokens 2048 \
--batch_size 64 \
--n_iterations 2 \
--warmup 3 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache \
--reuse_cache \
--use_flash_attention \
--trim_logits \
--limit_hpu_graphs \
--bucket_internal \
--bucket_size 256
```

To run generation with DeepSpeed-inference on 8 HPUs in 1 node, you should launch the script as follows:

```bash
cd optimum-habana/examples/text-generation

deepspeed --num_nodes 1 --num_gpus 8 --master_addr 127.0.0.1 --master_port 60008 run_generation.py \
--model_name_or_path /data/qwen1.5-72b \
--max_input_tokens 2048 \
--max_new_tokens 2048 \
--batch_size 128 \
--n_iterations 2 \
--warmup 3 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache \
--reuse_cache \
--use_flash_attention \
--trim_logits \
--limit_hpu_graphs \
--bucket_internal \
--bucket_size 512
```

Make sure `num_key_value_heads` in your model config should be divided by the # of HPUs.

For DeepSpeed inference on Qwen models larger than 72B with large batch size (>=64), it is recommended to use `bucket_size=512` for the inference, otherwise OOM issue may occur. For single card inference, `bucket_size=256` should work for most of the cases.

Currently Gaudi2 only supports BF16 and FP8 inference in this repo.Here is an example on the FP8 benchmark using the unit scale (scale = 1.0):

```bash
cd optimum-habana/examples/text-generation

QUANT_CONFIG="./quantization_config/unit_scale_quant.json" python3 run_generation.py \
--model_name_or_path /data/qwen1.5-7b \
--max_input_tokens 2048 \
--max_new_tokens 2048 \
--batch_size 128 \
--n_iterations 2 \
--warmup 3 \
--bf16 \
--fp8 \
--use_hpu_graphs \
--use_kv_cache \
--reuse_cache \
--attn_softmax_bf16 \
--trim_logits \
--limit_hpu_graphs \
--bucket_internal \
--bucket_size 256
```

Please note that currently the flash attention does not support in FP8 mode. If you enable `--use_flash_attention` on FP8 mode, you may encounter the performance issue.

For DeepSpeed inference, use the same `QUANT_CONFIG` environment in the command.

### Training

You may use the benchmark scripts in `sft` folder to perform full-parameter finetune or lora:
```bash
cd optimum-habana/examples/text-generation/Qwen/sft

bash run_finetune_HPU.sh 
```

or 

```bash
bash run_lora_HPU.sh
```
