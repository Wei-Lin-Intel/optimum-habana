# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import habana_frameworks.torch.hpex.experimental.transformer_engine as te
from habana_frameworks.torch.hpex.experimental.transformer_engine.distributed import activation_checkpointing

import deepspeed
import peft
from optimum.habana.transformers.models.llama.modeling_llama import GaudiLlamaForCausalLM

def unwrap_model(model):
    if isinstance(model, deepspeed.runtime.engine.DeepSpeedEngine):
        return unwrap_model(model.module)
    elif isinstance(model, peft.peft_model.PeftModelForCausalLM):
        return unwrap_model(model.base_model)
    elif isinstance(model, peft.tuners.lora.model.LoraModel):
        return unwrap_model(model.model)
    else:
        return model
        
class FP8ForwardMaker:
    def __init__(self, module, fp8_recipe_handler, use_activation_checkpointing=False):
        self.original_forward = module.forward
        self.fp8_forward = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe_handler)(module.forward)
        if use_activation_checkpointing:
            self.fp8_forward = activation_checkpointing()(self.fp8_forward)
        self.module = module
        module.forward = self.forward

    def forward(self, *args, **kwargs):
        if self.module.training:
            return self.fp8_forward(*args, **kwargs)
        else:
            return self.original_forward(*args, **kwargs)

    @staticmethod
    def convert(module, fp8_recipe_handler):
        model = unwrap_model(module)

        if model.is_gradient_checkpointing:

            if isinstance(model, GaudiLlamaForCausalLM):
                for _layer in model.model.layers:
                    FP8ForwardMaker(_layer, fp8_recipe_handler, use_activation_checkpointing=True)
                FP8ForwardMaker(model.lm_head, fp8_recipe_handler)
                return
    
        FP8ForwardMaker(module, fp8_recipe_handler, False)