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

# Inference on Mixtral 8x7B Model

## Requirements

Step1, use the docker image of Synapse 1.15.1 [Reference](https://docs.habana.ai/en/v1.15.1/) because this repo is based on this version. It is recommended to use Ubuntu 22.04 image:
```bash
docker pull vault.habana.ai/gaudi-docker/1.15.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest
``` 

Step2, you should install this optimum-habana:
```bash
pip install git+https://github.com/Wei-Lin-Intel/optimum-habana.git@v1.11-release
```

Step3, before you install HabanaAI DeepSpeed [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html), please replace `auto_tp.py` in the folder `deepspeed/module_inject`:
```bash
git clone https://github.com/HabanaAI/DeepSpeed.git -b 1.15.1
cd DeepSpeed
cp /your_path/optimum-habana/examples/text-generation/Mixtral/auto_tp.py deepspeed/module_inject
python3 setup.py bdist_wheel
cd dist && pip install deepspeed-0.12.4+hpu.synapse.v1.15.1-py3-none-any.whl 
```

## Inference

Please refer Mixtral-related instruction of `README.md` in the upper level directory.
