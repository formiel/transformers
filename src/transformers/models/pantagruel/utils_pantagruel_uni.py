# coding=utf-8
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import numpy as np
from collections import namedtuple
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])


def gather_unmasked(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep,
    )


def gather_unmasked_mask(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep[..., 0],  # ignore the feature dimension
    )


def masked_alibi(alibi_bias, mask_info):
    H = alibi_bias.size(1)

    orig_bias = alibi_bias

    index = mask_info.ids_keep.unsqueeze(1)[..., 0].unsqueeze(-1)
    alibi_bias = torch.gather(
        orig_bias,
        dim=-2,
        index=index.expand(-1, H, -1, mask_info.ids_restore.size(1)),
    )
    alibi_bias = torch.gather(
        alibi_bias,
        dim=-1,
        index=index.transpose(-1, -2).expand(-1, H, alibi_bias.size(-2), -1),
    )

    return alibi_bias


def random_masking(x, mask_ratio, mask_seed: Optional[MaskSeed]):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    generator = None
    if mask_seed is not None:
        seed = int(
            hash((mask_seed.seed, mask_seed.update, mask_seed.ids.sum().item())) % 1e6
        )
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    noise = torch.rand(N, L, generator=generator, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
    ids_restore = ids_shuffle.argsort(dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_unmasked = torch.gather(x, dim=1, index=ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], dtype=x.dtype, device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, D)

    return MaskInfo(
        x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep
    )


def get_alibi(
    max_positions: int,
    attention_heads: int,
    dims: int = 1,
    distance: str = "manhattan",
):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))

    if dims == 1:
        # prepare alibi position linear bias. Note that wav2vec2 is non
        # autoregressive model so we want a symmetric mask with 0 on the
        # diagonal and other wise linear decreasing valuees
        pos_bias = (
            torch.abs(
                torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
            )
            * -1
        )
    elif dims == 2:
        if distance == "manhattan":
            df = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        elif distance == "euclidean":
            df = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        n = math.sqrt(max_positions)
        assert n.is_integer(), n
        n = int(n)

        pos_bias = torch.zeros((max_positions, max_positions))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        new_x = i * n + j
                        new_y = k * n + l
                        pos_bias[new_x, new_y] = -df(i, j, k, l)

    else:
        raise Exception(f"unsupported number of alibi dims: {dims}")

    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )

    return alibi_bias


def get_alibi_bias(
    alibi_biases,
    batch_size,
    time_steps,
    heads,
    dtype,
    device,
    dims=1,
    distance="manhattan",
):
    cache_key = f"{dims}_{heads}_{distance}"

    buffered = alibi_biases.get(cache_key, None)

    target_size = heads * batch_size
    if (
        buffered is None
        or buffered.size(0) < target_size
        or buffered.size(1) < time_steps
        or buffered.dtype != dtype
        or buffered.device != device
    ):
        bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
        bn = max(target_size, buffered.size(0) if buffered is not None else 0) // heads

        buffered = (
            get_alibi(bt, heads, dims=dims, distance=distance)
            .to(dtype=dtype, device=device)
            .repeat(bn, 1, 1)
        )

        alibi_biases[cache_key] = buffered

    b = buffered[:target_size, :time_steps, :time_steps]
    b = b.view(batch_size, heads, time_steps, time_steps)
    return b


def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length)
                    + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + rng.random()
            )
            num_mask = max(min_masks, num_mask)
        else:
            raise ValueError()

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError(f"this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask


def _learned_alibi_bias(
    alibi_bias,
    batch_size,
    time_steps,
    heads,
    scale,
    dtype,
    device,
):
    assert alibi_bias.size(1) == heads, alibi_bias.shape
    assert alibi_bias.dtype == dtype, alibi_bias.dtype
    assert alibi_bias.device == device, alibi_bias.device

    if alibi_bias.size(-1) < time_steps:
        psz = math.ceil((time_steps - alibi_bias.size(-1)) / 2)
        alibi_bias = F.pad(alibi_bias, (psz, psz, psz, psz), mode="replicate")

    alibi_bias = alibi_bias.expand(batch_size, -1, -1, -1) * scale
    return alibi_bias[..., :time_steps, :time_steps]

def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx