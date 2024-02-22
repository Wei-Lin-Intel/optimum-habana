# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import math
import os
from typing import Optional, Tuple, Union

import torch
from torch import nn

#from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import logging


logger = logging.get_logger(__name__)
def gaudi_albert_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[BaseModelOutputWithPooling, Tuple]:
    """
    Same as https://github.com/huggingface/transformers/blob/a9eee2ffecc874df7dd635b2c6abb246fdb318cc/src/transformers/models/albert/modeling_albert.py#L689
    except that mixed precision is disabled for computing:
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    # torch.finfo must take the dtype of encoder_extended_attention_mask
    extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # bf16 compatibility
    extended_attention_mask = 1.0 - extended_attention_mask
    with torch.autocast(enabled=False, device_type="hpu"):
        extended_attention_mask = extended_attention_mask * torch.finfo(extended_attention_mask.dtype).min
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    embedding_output = self.embeddings(
        input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
    )
    encoder_outputs = self.encoder(
        embedding_output,
        extended_attention_mask,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = encoder_outputs[0]

    pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

def gaudi_albert_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    output_attentions: bool = False,
) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Same as https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/albert/modeling_albert.py#359
    except that dropout layer is not called if value is 0, otherwise move to CPU when it's not LAZY mode
    to avoid outofmemory issue.
    """

    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        seq_length = hidden_states.size()[1]
        position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    from optimum.habana.utils import get_hpu_memory_stats
    mem = get_hpu_memory_stats()
    for k, v in mem.items():
        logger.info("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
        break

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    if os.environ.get("PT_HPU_LAZY_MODE", "1") == "0":
        if self.attention_dropout.p != 0:
            attention_probs = attention_probs.to("cpu") #THIS SEEMS STILL CRASH AT THIS POINT.
            attention_probs = self.attention_dropout(attention_probs)
            attention_probs = attention_probs.to("hpu")
    else:
        attention_probs = self.attention_dropout(attention_probs)


    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.transpose(2, 1).flatten(2)

    projected_context_layer = self.dense(context_layer)
    if os.environ.get("PT_HPU_LAZY_MODE", "1") == "0":
        if self.output_dropout.p != 0:
            projected_context_layer = projected_context_layer.to("cpu")
            projected_context_layer_dropout = self.output_dropout(projected_context_layer)
            projected_context_layer_dropout = projected_context_layer_dropout.to("hpu")
        else:
            projected_context_layer_dropout = projected_context_layer
    else:
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)

    layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)

