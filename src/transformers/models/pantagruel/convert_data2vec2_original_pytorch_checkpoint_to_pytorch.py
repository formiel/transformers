# coding=utf-8
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert data2vec 2 checkpoint."""

import os
import argparse

import torch

from fairseq import tasks
from fairseq import utils
from fairseq import checkpoint_utils

from datasets import load_dataset
from transformers import AutoTokenizer

import torch.nn.functional as F

from configuration_data2vec2 import Data2Vec2MultiConfig
from modeling_data2vec2 import Data2Vec2MultiModel

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1
MASK_TOKEN, MASK_TOKEN_ID = "<mask>", 4
SPECIAL_TOKENS = [
            BOS_TOKEN,
            PAD_TOKEN,
            EOS_TOKEN,
            UNK_TOKEN,
            MASK_TOKEN,
        ]

FAIRSEQ = "/linkhome/rech/genlig01/ucy22cr/pantagruel/code/h100_py39_cu1241_torch251/fairspeech"
SAMPLE_TEXT = "Bonjour le monde !!"


def compare_tensors(tensor_a, tensor_b):
    max_absolute_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
    if max_absolute_diff > 0.0:
        raise ValueError(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(tensor_a, tensor_b, atol=1e-3)
    if not success:
        raise ValueError("!!!Something went wrong!!!")
    

def compare_outputs(input_values, fairseq_model, hf_model, mode, padding_mask=None):
    print(f"Forward using fairseq model...")
    fairseq_output = fairseq_model(
        source=input_values, mask=False, padding_mask=padding_mask, features_only=True
    )
    print(f"Forward using HF model...")
    hf_output = hf_model(
        input_values, padding_mask=padding_mask, mode=mode
    )

    print(f"Comparing outputs...")
    compare_tensors(hf_output.last_hidden_state, fairseq_output["x"])
    print(f'MATCHED!')
    print("*"*100)


@torch.no_grad()
def convert_data2vec2_checkpoint(args):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    checkpoint_path = args.checkpoint_path
    pytorch_dump_folder_path = args.pytorch_dump_folder_path

    fairseq_ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = fairseq_ckpt["model"]
    fairseq_model_config = fairseq_ckpt["cfg"]["model"]
    model_config = {k: v for k, v in fairseq_model_config.items() if not "ema" in k and not "decoder" in k and not "loss" in k}

    if args.vocab_dir is not None:
        # loading text model
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path, {})
        pretrained_args = state.get("cfg", None)
        assert pretrained_args is not None
        pretrained_args.criterion = None
        pretrained_args.lr_scheduler = None

        task = tasks.setup_task(pretrained_args.task, from_checkpoint=True)
        model_config["modalities"]["text"]["vocab_size"] = len(task.source_dictionary)
        print(f"Vocab size: {len(task.source_dictionary)}")

    # configuration
    configuration = Data2Vec2MultiConfig()
    configuration.update(model_config)

    # pre-trained weights
    hf_model = Data2Vec2MultiModel(configuration)
    keys = list(state_dict.keys())
    for k in keys:
        if "ema" in k or "decoder" in k:
            del state_dict[k]
    hf_model.load_state_dict(state_dict, strict=True)

    # saving pretrained model and configuration
    print(f"Saving pre-trained configuration and pre-trained weights...")
    hf_model.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)

    # comparing with fairseq state dict
    print("Loading from pretrained folder ...")
    test_model = Data2Vec2MultiModel.from_pretrained(pytorch_dump_folder_path)
    test_model.eval()
    test_model.freeze_feature_encoder()
    for n, p in test_model.named_parameters():
        compare_tensors(p, state_dict[n])
    print(f"Weights matched!")


@torch.no_grad()
def test_converted_weights(args):
    checkpoint_path = args.checkpoint_path
    pytorch_dump_folder_path = args.pytorch_dump_folder_path

    # HuggingFace model
    hf_model = Data2Vec2MultiModel.from_pretrained(pytorch_dump_folder_path)
    print(f"Pre-trained weights loaded to HF model!")
    hf_model.eval()
    hf_model.freeze_feature_encoder()

    # fairseq checkpoint
    os.chdir(FAIRSEQ)
    print(f"Loading fairseq model...")
    utils.import_user_module(args)
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path, {})
    pretrained_args = state.get("cfg", None)
    assert pretrained_args is not None
    pretrained_args.criterion = None
    pretrained_args.lr_scheduler = None

    task = tasks.setup_task(pretrained_args.task, from_checkpoint=True)
    print(f"fairseq model args: {pretrained_args.model}")
    fairseq_model = task.build_model(pretrained_args.model, from_checkpoint=True)
    fairseq_model.load_state_dict(state["model"], strict=True)
    fairseq_model.remove_pretraining_modules(modality="TEXT" if args.vocab_dir is not None else "AUDIO")
    fairseq_model.eval()
    print(f"Pre-trained weights loaded to fairseq model!")
    mode = "TEXT" if args.vocab_dir is not None else "AUDIO"

    if args.vocab_dir is not None:
        # text model
        configuration = hf_model.config
        print(f"Comparing outputs with randomized tensors...")
        input_values = torch.randint(configuration.modalities.text.vocab_size - 1, (1, configuration.modalities.text.max_source_positions), dtype=torch.int64)
        compare_outputs(
            input_values, fairseq_model, hf_model, mode=mode
        )
        
        print(f"Comparing outputs for SAMPLE TEXT...")
        if not args.use_bytebpe:
            tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
            encoded_ids = tokenizer.encode(SAMPLE_TEXT)
        else:
            from transformers import RobertaTokenizerFast
            # Text_Base_fr_4GB_v0
            tokenizer = RobertaTokenizerFast(
                    vocab_file=f"{args.vocab_dir}/bpe-bytelevel-vocab.json",
                    merges_file=f"{args.vocab_dir}/bpe-bytelevel-merges.txt",
                    bos_token=BOS_TOKEN,
                    eos_token=EOS_TOKEN,
                    cls_token=BOS_TOKEN,
                    sep_token=EOS_TOKEN,
                    pad_token=PAD_TOKEN,
                    unk_token=UNK_TOKEN,
                    mask_token=MASK_TOKEN
                )
            encoded_ids = tokenizer.encode(SAMPLE_TEXT)

        print(f'{SAMPLE_TEXT}: {encoded_ids}')

        # add post processors
        if "camembert" not in args.vocab_dir:
            from tokenizers import processors
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0",
                pair=f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0 $B:1 {EOS_TOKEN}:1",
                special_tokens=[
                    (BOS_TOKEN, BOS_TOKEN_ID),
                    (EOS_TOKEN, EOS_TOKEN_ID),
                ],
            )
        else:
            print("No need to add post processor for camembert")
        # Save tokenizers v0.22.1
        tokenizer.save_pretrained(args.pytorch_dump_folder_path)
        print(f"TOKENIZER: {tokenizer}")
        
        input_values = torch.tensor(
            encoded_ids, dtype=torch.int64
        ).unsqueeze(0)
        compare_outputs(
            input_values, fairseq_model, hf_model, mode=mode
        )
    else:
        print(f"Comparing outputs with randomized tensors...")
        input_values = torch.randn((3, 320000), dtype=torch.float32)
        with torch.no_grad():
            normalized_input_values = F.layer_norm(input_values, input_values.size()[1:])
        compare_outputs(
            normalized_input_values, fairseq_model, hf_model, mode=mode
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--user-dir", default="examples/data2vec")
    parser.add_argument("--vocab-dir", default=None)
    parser.add_argument("--do_convert", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--use_bytebpe", action="store_true",
                        help="for converting vocabulary learned with ByteBPE")
    args = parser.parse_args()

    if args.do_convert:
        convert_data2vec2_checkpoint(args)
    if args.do_test:
        test_converted_weights(args)