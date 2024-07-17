# coding=utf-8
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright 2022 the HuggingFace Inc. team. All rights reserved.
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

# Copyright from Fairseq

""" PyTorch Data2Vec2 Multi model."""
import math
import warnings
from typing import Callable
from functools import partial

import numpy as np

from torch import nn

from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import (
    Wav2Vec2BaseModelOutput,
    TokenClassifierOutput
)
from .configuration_data2vec2_multi import (
    Data2Vec2MultiConfig,
    D2v2ModalityConfig,
)
from .modeling_data2vec2_base import (
    ModalitySpecificEncoder,
    AudioEncoder,
    TextEncoder,
    AltBlock,
)


class Data2Vec2MultiPreTrainedModel(PreTrainedModel):
    config_class = Data2Vec2MultiConfig
    base_model_prefix = "data2vec2_multi"

    # use init_bert_params from fairseq
    # copied from fairseq.modules.transformer_sentence_encoder.py
    def _init_weights(self, module):
        """Initialize the weights"""

        def normal_(data):
            # with FSDP, module params will be on CUDA, so we cast them back to CPU
            # so that the RNG is consistent with and without FSDP
            data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

        def _init(module):
            if isinstance(module, nn.Linear):
                normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.Embedding):
                normal_(module.weight.data)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            if isinstance(module, AltBlock):
                normal_(module.attn.proj.weight.data)
            # init strategy for audio encoder
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                if module.bias is not None:
                    module.bias.data.zero_()
                if module.weight is not None:
                    module.weight.data.fill_(1.0)
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                    nn.init.uniform_(module.bias, a=-k, b=k)

        if isinstance(module, nn.ModuleList):
            for _, mod in enumerate(module):
                _init(mod)
        else:
            _init(module)

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     pretrained_model_name_or_path,
    #     *model_args,
    #     **kwargs,
    # ):
    #     config = cls.config_class()
    #     config.from_pretrained(pretrained_model_name_or_path)
    #     print(f"Loading configuration from pre-trained model: {type(config)}")
    #     return super().from_pretrained(pretrained_model_name_or_path, 
    #                                    *model_args, 
    #                                    config, 
    #                                    **kwargs,)
        

class Data2Vec2MultiModel(Data2Vec2MultiPreTrainedModel):
    def __init__(self, config: Data2Vec2MultiConfig):
        super().__init__(config)
        self.config = config
        modalities_cfg = config.modalities
        self.modalities = [config.supported_modality]

        make_layer_norm = partial(
            nn.LayerNorm, eps=config.norm_eps, elementwise_affine=config.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                config.embed_dim if dim is None else dim,
                config.num_heads if heads is None else heads,
                config.mlp_ratio,
                qkv_bias=True,
                drop=config.encoder_dropout,
                attn_drop=config.attention_dropout,
                mlp_drop=config.activation_dropout,
                post_mlp_drop=config.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=config.layer_norm_first,
                ffn_targets=not config.end_of_block_targets,
            )
        
        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        for mod in self.modalities:
            mod_cfg = getattr(modalities_cfg, mod.lower())
            enc = self.make_modality_encoder(
                mod_cfg,
                config.embed_dim,
                make_block,
                make_layer_norm,
                config.layer_norm_first,
                self.alibi_biases,
            )
            self.modality_encoders[mod] = enc

        self.dropout_input = nn.Dropout(config.dropout_input)

        dpr = np.linspace(config.start_drop_path_rate, config.end_drop_path_rate, config.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(config.depth)])

        self.norm = None
        if config.layer_norm_first:
            self.norm = make_layer_norm(config.embed_dim)

        self.num_updates = 0

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        for mod in self.modalities:
            self.modality_encoders[mod]._freeze_parameters()

    def make_modality_encoder(
        self,
        cfg: D2v2ModalityConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases,
    ) -> ModalitySpecificEncoder:
        if cfg.type == "AUDIO":
            enc_cls = AudioEncoder
        elif cfg.type == "TEXT":
            enc_cls = TextEncoder
        else:
            raise Exception(f"unsupported modality {cfg.type}")

        return enc_cls(
            cfg,
            embed_dim,
            make_block,
            norm_layer,
            layer_norm_first,
            alibi_biases,
        )
    
    def forward(
        self,
        input_values=None, # audio input
        input_ids=None, # text input
        attention_mask=None,
        padding_mask=None,
        mask=False,
        mode=None,
        output_hidden_states=True,
        return_dict=True,
    ):
        if mode is None:
            mode = "TEXT" if input_ids is not None else "AUDIO"
        feature_extractor = self.modality_encoders[mode]
        extractor_out = feature_extractor(
            input_ids if input_ids is not None else input_values,
            padding_mask,
            mask,
            remove_masked=False,
            clone_batch=1,
            mask_seeds=None,
            precomputed_mask=None,
        )
        x = extractor_out["x"]
        extract_features = x

        # encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.config.layerdrop == 0
                or (np.random.random() > self.config.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
        if masked_padding_mask is not None:
            masked_padding_mask = masked_padding_mask[
                :, feature_extractor.modality_cfg.num_extra_tokens :
            ]

        if not return_dict:
            return tuple(
                v
                for v in [
                    x,
                    extract_features,
                    layer_results,
                ]
                if v is not None
            )
        
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=x,
            extract_features=extract_features,
            hidden_states=layer_results if output_hidden_states else None,
            attentions=None, # switch to manual implementation with fast=False in forward pass of AltAttention as pytorch's dspa does not output attention weights
        )
from typing import Optional, Union, Tuple
import torch
        
class Data2VecForTokenClassification(Data2Vec2MultiPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        print(self.num_labels, " labels are defined")
        self.data2vec_text = Data2Vec2MultiModel(config)
        #self.data2vec_text = Data2Vec2MultiModel.from_pretrained(config.name_or_path)
        '''classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )'''
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
    '''
    @add_start_docstrings_to_model_forward(DATA2VEC2MULTI_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )'''
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.data2vec_text(
            input_ids,
            mode="TEXT"
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None, #outputs.hidden_states,
            attentions=None #outputs.attentions,
        )