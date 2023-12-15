# DPO pipeline for the creation of StackLlaMa 2: a Stack exchange llama-v2-7b model

## Prerequisites

Install all the dependencies in the `requirements.txt`:

```
$ pip install -U -r requirements.txt
```

## Training

There were two main steps to the DPO training process:
1. Supervised fine-tuning of the base llama-v2-7b model to create llama-v2-7b-se:
    - `python ../../gaudi_spawn.py --world_size 8 --use_mpi sft_llama2.py --training_args.output_dir="sft_output" --training-args.report_to none`
1. Run the DPO trainer using the model saved by the previous step:
    - `python ../../gaudi_spawn.py --world_size 8 --use_mpi dpo_llama2.py --model_name_or_path="sft_output/final_merged_checkpoint" --output_dir="dpo_output" --report_to=none`


## Running the model

We can load the DPO-trained LoRA adaptors which were saved by the DPO training step and load them via:

```py
import torch
from peft import AutoPeftModelForCausalLM


model = AutoPeftModelForCausalLM.from_pretrained(
    "dpo_output",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)

model.generate(...)
```
