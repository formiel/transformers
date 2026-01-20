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

""" PantagruelUni model."""
import math
import warnings
from typing import Optional, Tuple, Dict, List, Callable, Any, Union
from functools import partial
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import PreTrainedModel
from transformers.utils import (
    ModelOutput, TransformersKwargs, auto_docstring
)
from transformers.activations import ACT2FN, gelu
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask, _prepare_4d_attention_mask_for_sdpa
)
from transformers.utils.generic import can_return_tuple
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import (
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    CausalLMOutput,
    XVectorOutput,
)
from transformers.utils import auto_docstring, is_peft_available
from .configuration_pantagruel_uni import (
    PantagruelUniConfig,
    PantagruelModalityConfig,
    PantagruelAudioConfig,
    PantagruelTextConfig,
)

from .utils_pantagruel_uni import (
    _learned_alibi_bias,
    gather_unmasked,
    gather_unmasked_mask,
    masked_alibi,
    random_masking,
    get_alibi_bias,
    compute_mask_indices,
    index_put,
    MaskInfo, MaskSeed,
    make_positions,
)


@dataclass
class PantagruelUniBaseModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None # output of the encoder-only model
    pooler_output: Optional[torch.FloatTensor] = None # pooled output for text tasks, which is the first token representation followed by a dense layer and activation function
    local_features: Optional[torch.FloatTensor] = None # features before the Transformer encoder
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


# copied from fairseq.modules.grad_multiply
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


# copied from fairseq.modules.transpose_last.py
class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)
    

# copied from fairseq.modules.layer_norm.py
class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


# Copied from fairseq.modules.fp32_group_norm.py
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


# Copied from fairseq.modules.same_pad.py
class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


# Copied from fairseq.models.wav2vec.wav2vec2.py
class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x
    

# copied from fairseq.examples.data2vec.models.modalities.modules
class AltAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cosine_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop = proj_drop

        self.cosine_attention = cosine_attention

        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

    def forward(self, x, padding_mask=None, alibi_bias=None, fast=True):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # qkv x B x H x L x D
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        dtype = q.dtype

        attn = None
        if not fast:
            if self.cosine_attention:
                # cosine attention
                attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
                logit_scale = torch.clamp(
                    self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
                ).exp()
                attn = attn * logit_scale
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1) # B x C//H x L x L

            if alibi_bias is not None:
                attn = attn.type_as(alibi_bias)
                attn[:, : alibi_bias.size(1)] += alibi_bias

            if padding_mask is not None and padding_mask.any():
                attn = attn.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )

            attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
            # attn = self.attn_drop(attn)
            attn = F.dropout(attn, p=self.attn_drop if self.training else 0.0)
            x = (attn @ v).transpose(1, 2)
        else:
            # Using pytorch 2's sdpa
            assert not self.cosine_attention, "Not support cosine attention yet"
            # Integrate padding_mask and alibi_bias
            if padding_mask is not None and padding_mask.any():
                if alibi_bias is not None:
                    padding_mask = alibi_bias.masked_fill(
                            padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                            float("-inf"),
                        ).to(dtype=dtype)
                else:
                    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).to(
                        torch.bool).to(dtype=dtype)
            else:
                if alibi_bias is not None:
                    padding_mask = alibi_bias.to(dtype=dtype)
                else:
                    padding_mask = None

            x = F.scaled_dot_product_attention(q, k, v, 
                                attn_mask=padding_mask, 
                                dropout_p=self.attn_drop if self.training else 0.0,
                                scale=self.scale).transpose(1, 2)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = F.dropout(x, p=self.proj_drop if self.training else 0.0)

        return x, attn
    

# copied from fairseq.examples.data2vec.models.modalities.modules.py
class AltBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        post_mlp_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first=True,
        ffn_targets=False,
        cosine_attention=False,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets

        from timm.models.vision_transformer import DropPath, Mlp

        self.norm1 = norm_layer(dim)
        self.attn = AltAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

    def forward(self, x, padding_mask=None, alibi_bias=None, fast=True):
        if self.layer_norm_first:
            _x, _attn = self.attn(self.norm1(x), padding_mask, alibi_bias, fast=fast)
            x = x + self.drop_path(_x)
            r = x = self.mlp(self.norm2(x))
            t = x
            x = r + self.drop_path(self.post_mlp_dropout(x))
            if not self.ffn_targets:
                t = x
        else:
            _x, _attn = self.attn(x, padding_mask, alibi_bias, fast=fast)
            x = x + self.drop_path(_x)
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
            if not self.ffn_targets:
                t = x

        return x, t, _attn


# copied from fairseq.data2vec.models.modalities.modules
class BlockEncoder(nn.Module):
    def __init__(self, blocks, norm_layer, layer_norm_first, layerdrop, dropout):
        super().__init__()
        self.blocks = blocks
        self.norm = norm_layer
        self.layer_norm_first = layer_norm_first
        self.layerdrop = layerdrop
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, padding_mask, alibi_bias, alibi_scale):
        if self.norm is not None and not self.layer_norm_first:
            x = self.norm(x)

        x = self.dropout(x)

        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.layerdrop == 0
                or (np.random.random() > self.layerdrop)
            ):
                ab = alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)
                x, _, _ = blk(x, padding_mask, ab)

        if self.norm is not None and self.layer_norm_first:
            x = self.norm(x)

        return x
    

# copied from fairseq.examples.data2vec.models
class ModalitySpecificEncoder(nn.Module):
    def __init__(
        self,
        modality_cfg: PantagruelModalityConfig,
        embed_dim: int,
        local_encoder: nn.Module,
        project_features: nn.Module,
        fixed_positional_encoder: Optional[nn.Module],
        relative_positional_encoder: Optional[nn.Module],
        context_encoder: nn.Module,
        decoder: nn.Module,
        get_alibi_bias: Optional[Callable[[int, int, str, str], torch.Tensor]],
    ):
        super().__init__()

        self.modality_cfg = modality_cfg
        self.local_encoder = local_encoder
        self.project_features = project_features
        self.fixed_positional_encoder = fixed_positional_encoder
        self.relative_positional_encoder = relative_positional_encoder
        self.context_encoder = context_encoder

        self.decoder = None
        self.get_alibi_bias = get_alibi_bias if modality_cfg.use_alibi_encoder else None

        self.local_grad_mult = self.modality_cfg.local_grad_mult

        self.extra_tokens = None
        if modality_cfg.num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.zeros(1, modality_cfg.num_extra_tokens, embed_dim)
            )
            if not modality_cfg.init_extra_token_zero:
                nn.init.normal_(self.extra_tokens)
            elif self.extra_tokens.size(1) > 1:
                nn.init.normal_(self.extra_tokens[:, 1:])

        self.alibi_scale = None
        if self.get_alibi_bias is not None:
            self.alibi_scale = nn.Parameter(
                torch.full(
                    (
                        (modality_cfg.prenet_depth + modality_cfg.model_depth)
                        if modality_cfg.learned_alibi_scale_per_layer
                        else 1,
                        1,
                        self.modality_cfg.num_alibi_heads
                        if modality_cfg.learned_alibi_scale_per_head
                        else 1,
                        1,
                        1,
                    ),
                    modality_cfg.alibi_scale,
                    dtype=torch.float,
                ),
                requires_grad=modality_cfg.learned_alibi_scale,
            )

        if modality_cfg.learned_alibi and self.get_alibi_bias is not None:
            assert modality_cfg.alibi_max_pos is not None
            alibi_bias = self.get_alibi_bias(
                batch_size=1,
                time_steps=modality_cfg.alibi_max_pos,
                heads=modality_cfg.num_alibi_heads,
                scale=1.0,
                dtype=torch.float,
                device="cpu",
            )
            self.alibi_bias = nn.Parameter(alibi_bias)
            self.get_alibi_bias = partial(
                _learned_alibi_bias, alibi_bias=self.alibi_bias
            )

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder._freeze_parameters
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def convert_padding_mask(self, x, padding_mask):
        return padding_mask

    def local_features(self, features):
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.local_encoder(features)
            else:
                x = GradMultiply.apply(
                    self.local_encoder(features), self.local_grad_mult
                )
        else:
            with torch.no_grad():
                x = self.local_encoder(features)

        x = self.project_features(x)
        return x
    
    def contextualized_features(
        self,
        x,
        padding_mask,
        mask,
        remove_masked,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):

        if padding_mask is not None:
            padding_mask = self.convert_padding_mask(x, padding_mask)

        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        orig_B, orig_T, _ = x.shape
        pre_mask_B = orig_B
        mask_info = None

        x_pos = None
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, padding_mask)

        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0)
                if mask_seeds is not None:
                    clone_hash = [
                        int(hash((mask_seeds.seed, ind)) % 1e10)
                        for ind in range(clone_batch - 1)
                    ]
                    clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                    id = mask_seeds.ids
                    id = id.repeat_interleave(clone_batch, 0)
                    id = id.view(-1, clone_batch) + clone_hash.to(id)
                    id = id.view(-1)
                    mask_seeds = MaskSeed(
                        seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                    )
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

            x, mask_info = self.compute_mask(
                x,
                padding_mask,
                mask_seed=mask_seeds,
                apply=self.relative_positional_encoder is not None or not remove_masked,
                precomputed_mask=precomputed_mask,
            )

        if self.relative_positional_encoder is not None:
            x_pos = self.relative_positional_encoder(x)

        masked_padding_mask = padding_mask
        if mask and remove_masked:
            x = mask_info.x_unmasked
            if x_pos is not None:
                x = x + gather_unmasked(x_pos, mask_info)

            if padding_mask is not None and padding_mask.any():
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None

        elif x_pos is not None:
            x = x + x_pos

        alibi_bias = None
        alibi_scale = self.alibi_scale

        if self.get_alibi_bias is not None:
            alibi_bias = self.get_alibi_bias(
                batch_size=pre_mask_B,
                time_steps=orig_T,
                heads=self.modality_cfg.num_alibi_heads,
                dtype=torch.float32,
                device=x.device,
            )

            if alibi_scale is not None:
                alibi_scale = alibi_scale.clamp_min(0)
                if alibi_scale.size(0) == 1:
                    alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                    alibi_scale = None

            if clone_batch > 1:
                alibi_bias = alibi_bias.repeat_interleave(clone_batch, 0)

            if mask_info is not None and remove_masked:
                alibi_bias = masked_alibi(alibi_bias, mask_info)

        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            if masked_padding_mask is not None:
                # B x T
                masked_padding_mask = F.pad(masked_padding_mask, (num, 0))
            if alibi_bias is not None:
                # B x H x T x T
                alibi_bias = F.pad(alibi_bias, (num, 0, num, 0))

        x = self.context_encoder(
            x,
            masked_padding_mask,
            alibi_bias,
            alibi_scale[: self.modality_cfg.prenet_depth]
            if alibi_scale is not None
            else None,
        )

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "alibi_bias": alibi_bias,
            "alibi_scale": alibi_scale[self.modality_cfg.prenet_depth :]
            if alibi_scale is not None and alibi_scale.size(0) > 1
            else alibi_scale,
            "encoder_mask": mask_info,
        }

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        x = self.local_features(features)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )
    
    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        precomputed_mask,
    ):
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self.make_maskinfo(x, mask)
        else:
            B, T, C = x.shape
            cfg = self.modality_cfg

            mask_prob = cfg.mask_prob

            if (
                cfg.mask_prob_min is not None
                and cfg.mask_prob_min >= 0
                and cfg.mask_prob_min < mask_prob
            ):
                mask_prob = np.random.uniform(cfg.mask_prob_min, mask_prob)

            if mask_prob > 0:
                if cfg.mask_length == 1:
                    mask_info = random_masking(x, mask_prob, mask_seed)
                else:
                    if self.modality_cfg.inverse_mask:
                        mask_prob = 1 - mask_prob

                    mask = compute_mask_indices(
                        (B, T),
                        padding_mask,
                        mask_prob,
                        cfg.mask_length,
                        min_masks=1,
                        require_same_masks=True,
                        mask_dropout=cfg.mask_dropout,
                        add_masks=cfg.add_masks,
                        seed=mask_seed.seed if mask_seed is not None else None,
                        epoch=mask_seed.update if mask_seed is not None else None,
                        indices=mask_seed.ids if mask_seed is not None else None,
                    )

                    mask = torch.from_numpy(mask).to(device=x.device)
                    if self.modality_cfg.inverse_mask:
                        mask = 1 - mask
                    mask_info = self.make_maskinfo(x, mask)
            else:
                mask_info = None

        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def make_maskinfo(self, x, mask, shape=None):
        if shape is None:
            B, T, D = x.shape
        else:
            B, T, D = shape

        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)

        len_keep = T - mask[0].sum()
        if self.modality_cfg.keep_masked_pct > 0:
            len_keep += round((T - int(len_keep)) * self.modality_cfg.keep_masked_pct)

        ids_keep = ids_shuffle[:, :len_keep]

        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x_unmasked = torch.gather(x, dim=1, index=ids_keep)

        mask_info = MaskInfo(
            x_unmasked=x_unmasked,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )
        return mask_info

    def apply_mask(self, x, mask_info):
        cfg = self.modality_cfg
        B, T, C = x.shape

        if mask_info is not None:
            mask = mask_info.mask
            if cfg.encoder_zero_mask:
                x = x * (1 - mask.type_as(x).unsqueeze(-1))
            else:
                num_masks = mask.sum().item()
                masks = x.new_empty(num_masks, x.size(-1)).normal_(
                    0, cfg.mask_noise_std
                )
                x = index_put(x, mask, masks)
        if cfg.mask_channel_prob > 0:
            mask_channel = compute_mask_indices(
                (B, C),
                None,
                cfg.mask_channel_prob,
                cfg.mask_channel_length,
            )
            mask_channel = (
                torch.from_numpy(mask_channel)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x = index_put(x, mask_channel, 0)
        return x
    

# copied from fairseq.examples.data2vec.models.modalities.audio
class AudioEncoder(ModalitySpecificEncoder):

    modality_cfg: PantagruelAudioConfig

    def __init__(
        self,
        modality_cfg: PantagruelAudioConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
    ):

        self.feature_enc_layers = eval(modality_cfg.feature_encoder_spec)
        feature_embed_dim = self.feature_enc_layers[-1][0]

        local_encoder = ConvFeatureExtractionModel(
            conv_layers=self.feature_enc_layers,
            dropout=0.0,
            mode=modality_cfg.extractor_mode,
            conv_bias=False,
        )

        project_features = nn.Sequential(
            TransposeLast(),
            nn.LayerNorm(feature_embed_dim),
            nn.Linear(feature_embed_dim, embed_dim),
        )

        num_pos_layers = modality_cfg.conv_pos_depth
        k = max(3, modality_cfg.conv_pos_width // num_pos_layers)

        positional_encoder = nn.Sequential(
            TransposeLast(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=modality_cfg.conv_pos_groups,
                    ),
                    SamePad(k),
                    TransposeLast(),
                    LayerNorm(embed_dim, elementwise_affine=False),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(num_pos_layers)
            ],
            TransposeLast(),
        )

        if modality_cfg.conv_pos_pre_ln:
            positional_encoder = nn.Sequential(LayerNorm(embed_dim), positional_encoder)

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )
        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        decoder = None

        alibi_bias_fn = partial(get_alibi_bias, alibi_biases=alibi_biases)

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=None,
            relative_positional_encoder=positional_encoder,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def convert_padding_mask(self, x, padding_mask):
        def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """

            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length - kernel_size) / stride + 1)

            for i in range(len(self.feature_enc_layers)):
                input_lengths = _conv_out_length(
                    input_lengths,
                    self.feature_enc_layers[i][1],
                    self.feature_enc_layers[i][2],
                )

            return input_lengths.to(torch.long)

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = get_feat_extract_output_lengths(input_lengths)

            if padding_mask.any():
                padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (
                    1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
                ).bool()
            else:
                padding_mask = torch.zeros(
                    x.shape[:2], dtype=torch.bool, device=x.device
                )

        return padding_mask
    

# copied from fairseq
class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = make_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace
                )
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


# copied from fairseq
class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.register_buffer("weights", SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        ), persistent=False)
        self.max_positions = int(1e5)
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Ignore some deprecated keys that were used in older versions
        deprecated_keys = ["weights", "_float_tensor"]
        for key in deprecated_keys:
            if prefix + key in state_dict:
                del state_dict[prefix + key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            ).to(self.weights)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )


# copied from fairseq.modules
def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1,
        )
    return m


# copied from fairseq.examples.data2vec.modules
class TextLocalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        max_source_positions,
        pad_idx,
        no_scale_embedding,
        layernorm_embedding,
        dropout,
        no_token_positional_embeddings,
        learned_pos,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.dropout_module = nn.Dropout(dropout)

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, pad_idx)
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                max_source_positions,
                embed_dim,
                pad_idx,
                learned=learned_pos,
            )
            if not no_token_positional_embeddings
            else None
        )
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(embed_dim)

        self.layernorm_embedding = None
        if layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)

    def forward(self, src_tokens):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x
    

class TextEncoder(ModalitySpecificEncoder):

    modality_cfg: PantagruelTextConfig

    def __init__(
        self,
        modality_cfg: PantagruelTextConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
    ):
        self.pad_idx = modality_cfg.pad_token_id
        self.vocab_size = modality_cfg.vocab_size

        local_encoder = TextLocalEncoder(
            vocab_size=self.vocab_size,
            embed_dim=embed_dim,
            max_source_positions=modality_cfg.max_source_positions,
            pad_idx=self.pad_idx,
            no_scale_embedding=modality_cfg.no_scale_embedding,
            layernorm_embedding=modality_cfg.layernorm_embedding,
            dropout=modality_cfg.dropout,
            no_token_positional_embeddings=modality_cfg.no_token_positional_embeddings,
            learned_pos=modality_cfg.learned_pos,
        )
        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )
        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim)
            if not layer_norm_first and modality_cfg.prenet_depth > 0
            else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout if modality_cfg.prenet_depth > 0 else 0.0,
        )
        decoder = None

        alibi_bias_fn = partial(get_alibi_bias, alibi_biases=alibi_biases)

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=nn.Identity(),
            fixed_positional_encoder=None,
            relative_positional_encoder=None,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def convert_padding_mask(self, x, padding_mask):
        if padding_mask is None or padding_mask.size(1) == x.size(1):
            return padding_mask

        diff = self.downsample - padding_mask.size(1) % self.downsample
        if 0 < diff < self.downsample:
            padding_mask = F.pad(padding_mask, (0, diff), value=True)

        padding_mask = padding_mask.view(padding_mask.size(0), -1, self.downsample)
        padding_mask = padding_mask.all(-1)
        if padding_mask.size(1) > x.size(1):
            padding_mask = padding_mask[:, : x.size(1)]

        assert x.size(1) == padding_mask.size(
            1
        ), f"{x.size(1), padding_mask.size(1), diff, self.downsample}"

        return padding_mask


# copied from transformers.models.data2vec.modeling_data2vec.PantagruelUniTextPooler
class PantagruelUniTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# copied from transformers.models.data2vec.modeling_data2vec_audio
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss


# copied from transformers.models.data2vec.modeling_data2vec_audio
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_peft_available():
            from peft.tuners.lora import LoraLayer

        if is_peft_available():
            if isinstance(self.kernel, LoraLayer):
                warnings.warn(
                    "Detected LoRA on TDNNLayer. LoRA weights won't be applied due to optimization. "
                    "You should exclude TDNNLayer from LoRA's target modules.",
                )

        # for backward compatibility, we keep nn.Linear but call F.conv1d for speed up
        hidden_states = hidden_states.transpose(1, 2)
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.activation(hidden_states)
        return hidden_states


@auto_docstring
class PantagruelUniPreTrainedModel(PreTrainedModel):
    config_class = PantagruelUniConfig
    base_model_prefix = "pantagruel_uni"

    # use init_bert_params from fairseq
    # copied from fairseq.modules.transformer_sentence_encoder.py
    def _init_weights(self, module):
        """Initialize the weights"""

        def normal_(data):
            # with FSDP, module params will be on CUDA, so we cast them back to CPU
            # so that the RNG is consistent with and without FSDP
            if not data.is_meta:
                data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))
            return data

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

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.modalities.audio.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
        

@auto_docstring
class PantagruelUniModel(PantagruelUniPreTrainedModel):

    def __init__(
        self, config: PantagruelUniConfig, add_pooling_layer: bool = True
    ):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
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

        self.text_pooler = None
        if add_pooling_layer and config.supported_modality == "TEXT":
            self.text_pooler = PantagruelUniTextPooler(config)

        self.norm = None
        if config.layer_norm_first:
            self.norm = make_layer_norm(config.embed_dim)

        self.num_updates = 0

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        if "TEXT" in self.modality_encoders:
            return self.modality_encoders["TEXT"].local_encoder.embed_tokens

    def set_input_embeddings(self, value):
        if "TEXT" in self.modality_encoders:
            self.modality_encoders["TEXT"].local_encoder.embed_tokens = value

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

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        for mod in self.modalities:
            self.modality_encoders[mod]._freeze_parameters()
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

    def make_modality_encoder(
        self,
        cfg: PantagruelModalityConfig,
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
        token_type_ids=None,
        attention_mask=None,
        padding_mask=None,
        mask=False,
        mode=None,
        output_hidden_states=True,
        output_attn_weights=False,
        return_dict=True,
    ) -> Union[Tuple, PantagruelUniBaseModelOutput]:
        r"""
        Performs a forward pass of the model for either audio or text inputs.

        The modality is automatically inferred if `mode` is not provided:
            `"TEXT"` is used when `input_ids` is specified, otherwise `"AUDIO"`.
        
        Args:
            input_values (`torch.FloatTensor`, *optional*):
                Audio input values of shape `(batch_size, sequence_length)`
                containing *normalized* audio samples
                Required when operating in `"AUDIO"` mode.

            input_ids (`torch.LongTensor`, *optional*):
                Tokenized text input IDs of shape `(batch_size, sequence_length)`.
                Required when operating in `"TEXT"` mode.

            attention_mask (`torch.LongTensor`, *optional*):
                Attention mask for text inputs, with values in `{0, 1}`:
                - `1` for tokens that should be attended to,
                - `0` for tokens that should be masked.
                If provided and `padding_mask` is `None`, it will be converted internally
                to a padding mask.

            padding_mask (`torch.BoolTensor` or `torch.LongTensor`, *optional*):
                Padding mask indicating which positions are padded:
                - `1` (or `True`) for padded positions (not attended to),
                - `0` (or `False`) for non-padded positions.
                If not provided and `attention_mask` is given, this is inferred as
                the logical negation of `attention_mask`.

            mask (`bool`, *optional*, defaults to `False`):
                Whether to apply input masking.

            mode (`str`, *optional*):
                Explicitly specifies the input modality. Supported values are
                `"TEXT"` and `"AUDIO"`. If `None`, the mode is inferred from the
                provided inputs.

            output_hidden_states (`bool`, *optional*, defaults to `True`):
                Whether to return the hidden states of all layers.

            output_attn_weights (`bool`, *optional*, defaults to `False`):
                Whether to return attention weights.

            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`ModelOutput`] instead of a plain tuple.

        Returns:
            [`ModelOutput`] or `tuple`:
                The model outputs. If `return_dict=True`, a [`ModelOutput`] is returned
                containing (depending on configuration) the final hidden states,
                optional hidden states from all layers, and optional attention weights.
                If `return_dict=False`, a tuple is returned with the same contents in
                a fixed order.
        """

        if mode is None:
            mode = "TEXT" if input_ids is not None else "AUDIO"

        if padding_mask is None and attention_mask is not None:
            padding_mask = ~attention_mask.bool() # attention mask: 1 means to attend to (not masked), 0 means not to attend to (masked). padding mask: 1 means padded (not attend to), 0 means not padded (to attend to)

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
        local_features = x

        # encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        attn_weights = []
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

                x, lr, _attn = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                    fast=not output_attn_weights,
                )
                layer_results.append(lr)
                attn_weights.append(_attn)

        if self.norm is not None:
            x = self.norm(x)

        x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
        if masked_padding_mask is not None:
            masked_padding_mask = masked_padding_mask[
                :, feature_extractor.modality_cfg.num_extra_tokens :
            ]

        txt_pooled_output = (
            self.text_pooler(x) if self.text_pooler is not None else None
        )

        if not return_dict:
            return tuple(
                v
                for v in [
                    x,
                    txt_pooled_output,
                    local_features,
                    layer_results,
                    attn_weights,
                ]
                if v is not None
            )
        
        return PantagruelUniBaseModelOutput(
            last_hidden_state=x,
            pooler_output=txt_pooled_output,
            local_features=local_features,
            hidden_states=layer_results if output_hidden_states else None,
            attentions=attn_weights if output_attn_weights else None,
        )


class PantagruelTextLMHead(nn.Module):
    """PantagruelText Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)

        self.decoder = nn.Linear(config.embed_dim, config.modalities.text.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.modalities.text.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class PantagruelTextClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.encoder_dropout
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.embed_dim, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@auto_docstring
class PantagruelUniForMaskedLM(PantagruelUniPreTrainedModel):
    # _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `PantagruelTextForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.pantagruel_uni = PantagruelUniModel(config, add_pooling_layer=False)
        self.lm_head = PantagruelTextLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.pantagruel_uni(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            mask=False,
            mode="TEXT",
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


_HIDDEN_STATES_START_POSITION = 2


@auto_docstring(
    custom_intro="""
    PantagruelUniModel with a sequence classification or regression head on top (a linear layer applied to a pooled representation of the sequence).
    This model supports text and audio modalities. The classification head and internal processing are selected automatically based on the configuration.
    """
)
class PantagruelUniForSequenceClassification(PantagruelUniPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.pantagruel_uni = PantagruelUniModel(config, add_pooling_layer=False)

        if config.supported_modality == "TEXT":
            logger.info("Initializing PantagruelUniForSequenceClassification for TEXT")
            self.classifier = PantagruelTextClassificationHead(config)
        elif config.supported_modality == "AUDIO":
            logger.info("Initializing PantagruelUniForSequenceClassification for AUDIO")
            num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
            if config.modalities.audio.use_weighted_layer_sum:
                self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.projector = nn.Linear(config.hidden_size, config.modalities.audio.classifier_proj_size)
            self.classifier = nn.Linear(config.modalities.audio.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
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
        self.pantagruel_uni.freeze_feature_encoder()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.pantagruel_uni.parameters():
            param.requires_grad = False

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, SequenceClassifierOutput]:
        r"""
        Performs a forward pass for sequence classification or regression.

        This method supports both **text** and **audio** inputs. The modality is inferred
            from the provided inputs and the model configuration.
        
        Args:
            input_values (`torch.FloatTensor`, *optional*):
                Audio input values of shape `(batch_size, sequence_length)`
                containing *normalized* audio samples.
            input_ids (`torch.LongTensor`, *optional*):
                Tokenized text input IDs of shape `(batch_size, sequence_length)`.
                Used when the model is configured for `"TEXT"` modality.

            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. 
                    If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), 
                    If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if self.config.supported_modality == "TEXT":
            outputs = self.pantagruel_uni(
                input_ids=input_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                mask=False,
                mode="TEXT",
                return_dict=True,
            )

            sequence_output = outputs.last_hidden_state
            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                labels = labels.to(logits.device)

                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

        else:
            outputs = self.pantagruel_uni(
                input_values=input_values,
                attention_mask=attention_mask,
                mask=False,
                mode="AUDIO",
                output_hidden_states=output_hidden_states,
                output_attn_weights=output_attentions,
                return_dict=return_dict,
            )
            if self.config.modalities.audio.use_weighted_layer_sum:
                hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
                hidden_states = torch.stack(hidden_states, dim=1)
                norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
                hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            else:
                hidden_states = outputs[0]

            hidden_states = self.projector(hidden_states)
            if attention_mask is None:
                pooled_output = hidden_states.mean(dim=1)
            else:
                padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
                expand_padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
                hidden_states[~expand_padding_mask] = 0.0
                pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

            logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
                return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


@auto_docstring
class PantagruelUniForMultipleChoice(PantagruelUniPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.pantagruel_uni = PantagruelUniModel(config)
        self.dropout = nn.Dropout(config.encoder_dropout)
        self.classifier = nn.Linear(config.embed_dim, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MultipleChoiceModelOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.data2vec_text(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            padding_mask=flat_attention_mask,
            mask=False,
            mode="TEXT",
            return_dict=True,
        )
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(reshaped_logits.device)
            loss = loss_fct(reshaped_logits, labels)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class PantagruelUniForTokenClassification(PantagruelUniPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.pantagruel_uni = PantagruelUniModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.encoder_dropout
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.embed_dim, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.pantagruel_uni(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            mask=False,
            mode="TEXT",
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class PantagruelUniForQuestionAnswering(PantagruelUniPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.pantagruel_uni = PantagruelUniModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.embed_dim, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, QuestionAnsweringModelOutput]:

        outputs = self.pantagruel_uni(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            mask=False,
            mode="TEXT",
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class PantagruelUniForAudioFrameClassification(PantagruelUniPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        if hasattr(config.modalities.audio, "add_adapter") and config.modalities.audio.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Data2VecAudio adapters (config.add_adapter=True)"
            )

        self.pantagruel_uni = PantagruelUniModel(config, add_pooling_layer=False)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.modalities.audio.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
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
        self.pantagruel_uni.freeze_feature_encoder()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.pantagruel_uni.parameters():
            param.requires_grad = False

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, TokenClassifierOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
            (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
            To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and conversion
            into a tensor of type `torch.FloatTensor`. See [`Data2VecAudioProcessor.__call__`] for details.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            True if self.config.modalities.audio.use_weighted_layer_sum 
            else output_hidden_states
        )

        outputs = self.pantagruel_uni(
            input_values=input_values,
            attention_mask=attention_mask,
            mask=False,
            mode="AUDIO",
            output_hidden_states=output_hidden_states,
            output_attn_weights=output_attentions,
            return_dict=return_dict,
        )

        if self.config.modalities.audio.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    PantagruelUniForCTC Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
    """
)
class PantagruelUniForCTC(PantagruelUniPreTrainedModel):
    def __init__(self, config):
        r"""
        target_lang (`str`, *optional*):
            Language id of adapter weights. Adapter weights are stored in the format adapter.<lang>.safetensors or
            adapter.<lang>.bin. Only relevant when using an instance of [`Data2VecAudioForCTC`] with adapters. Uses 'eng' by
            default.
        """
        super().__init__(config)

        self.pantagruel_uni = PantagruelUniModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.modalities.audio.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.modalities.audio.output_hidden_size if hasattr(config.modalities.audio, "add_adapter") and config.modalities.audio.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.modalities.audio.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()
    
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
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
        self.pantagruel_uni.freeze_feature_encoder()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.pantagruel_uni.parameters():
            param.requires_grad = False

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and labels.max() >= self.config.modalities.audio.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.modalities.audio.vocab_size}")

        outputs = self.pantagruel_uni(
            input_values=input_values,
            attention_mask=attention_mask,
            mask=False,
            mode="AUDIO",
            output_hidden_states=output_hidden_states,
            output_attn_weights=output_attentions,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class PantagruelUniForXVector(PantagruelUniPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.pantagruel_uni = PantagruelUniModel(config, add_pooling_layer=False)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.modalities.audio.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.modalities.audio.tdnn_dim[0])

        tdnn_layers = [
            TDNNLayer(config.modalities.audio, i) for i in range(len(config.modalities.audio.tdnn_dim))
        ]
        self.tdnn = nn.ModuleList(tdnn_layers)

        self.feature_extractor = nn.Linear(
            config.modalities.audio.tdnn_dim[-1] * 2, config.modalities.audio.xvector_output_dim
        )
        self.classifier = nn.Linear(
            config.modalities.audio.xvector_output_dim, config.modalities.audio.xvector_output_dim
        )

        self.objective = AMSoftmaxLoss(
            config.modalities.audio.xvector_output_dim, config.num_labels
        )

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
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
        self.pantagruel_uni.freeze_feature_encoder()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.pantagruel_uni.parameters():
            param.requires_grad = False
    
    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.config.modalities.audio.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, XVectorOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
            (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
            To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and conversion
            into a tensor of type `torch.FloatTensor`. See [`Data2VecAudioProcessor.__call__`] for details.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.modalities.audio.use_weighted_layer_sum else output_hidden_states

        outputs = self.pantagruel_uni(
            input_values=input_values,
            attention_mask=attention_mask,
            mask=False,
            mode="AUDIO",
            output_hidden_states=output_hidden_states,
            output_attn_weights=output_attentions,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling
        if attention_mask is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)

        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "PantagruelUniForMaskedLM",
    "PantagruelUniForMultipleChoice",
    "PantagruelUniForQuestionAnswering",
    "PantagruelUniForSequenceClassification",
    "PantagruelUniForTokenClassification",
    "PantagruelUniModel",
    "PantagruelUniPreTrainedModel",
    "PantagruelUniForAudioFrameClassification",
]