import warnings
from typing import Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import torch
import torch.nn as nn

from transformers.utils import (
    logging,
)

from transformers.models.mamba.modeling_mamba import (
    MambaCache,
)

logger = logging.get_logger(__name__)


def gaudi_MambaForCausalLM_prepare_inputs_for_generation(
        self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        token_idx = kwargs.get("token_idx", None)
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            if token_idx is None:
                input_ids = input_ids[:, -1].unsqueeze(-1)
            else:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
        return model_inputs
