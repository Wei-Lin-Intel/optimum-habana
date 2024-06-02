# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import json
import logging
import os
import time

import lm_eval.evaluator
import lm_eval.tasks
import torch
from utils import initialize_model, finalize_quantization, setup_quantization
from habana_model_adapter import setup_lm_eval_parser, HabanaModelAdapter, print_results

from optimum.habana.utils import get_hpu_memory_stats

logger = logging.getLogger(__name__)

def main():
    args = setup_lm_eval_parser()
    # prepare INC API call for the module is done in initialize_model to enable quantization
    model, tokenizer, generation_config = initialize_model(args, logger)

    lm_tasks = lm_eval.tasks.get_task_dict(args.tasks)
    with torch.no_grad():
        lm = HabanaModelAdapter(tokenizer, model, args, generation_config)

    # measurement step is done first to calibrate the quantization values
    eval_start_measure = time.perf_counter()
    # running data on high-precision model for calibration
    results_msr = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=args.limit_iters)
    if args.device == "hpu":
        import habana_frameworks.torch.hpu as torch_hpu
        torch_hpu.synchronize()
    eval_end_measure = time.perf_counter()
    results_msr["args"] = vars(args)
    results_msr["duration"] = eval_end_measure - eval_start_measure
    print_results(args, results_msr)

    if args.quant_config:
        # saving the measurement of the calibration data
        finalize_quantization(lm.model)
        # convert INC API is called to quantize the module in setup_quantization
        model = setup_quantization(lm.model)

    eval_start = time.perf_counter()
    # running data on quantized model
    results = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=args.limit_iters)
    if args.device == "hpu":
        import habana_frameworks.torch.hpu as torch_hpu
        torch_hpu.synchronize()
    eval_end = time.perf_counter()

    results["args"] = vars(args)
    results["duration_quant"] = eval_end - eval_start

    print_results(args, results)

    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization_path)


if __name__ == "__main__":
    main()
