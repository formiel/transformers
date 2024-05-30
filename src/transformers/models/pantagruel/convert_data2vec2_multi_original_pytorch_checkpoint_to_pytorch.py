# coding=utf-8
# Modified by Hang Le (hangtp.le@gmail.com)
# Original copyrights below
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
from tokenizers import ByteLevelBPETokenizer

from transformers import (
    Wav2Vec2Processor,
    Data2Vec2MultiConfig,
    Data2Vec2MultiModel,
)
UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1

FAIRSEQ = "/linkhome/rech/genlig01/umz16dj/code/fairspeech"
SAMPLE_TEXT = "Bonjour le monde !!"


def compare_tensors(tensor_a, tensor_b):
    max_absolute_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(tensor_a, tensor_b, atol=1e-3)
    if not success:
        raise ValueError("!!!Something went wrong!!!")


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
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path, {})
        w2v_args = state.get("cfg", None)
        assert w2v_args is not None
        w2v_args.criterion = None
        w2v_args.lr_scheduler = None
        task = tasks.setup_task(w2v_args.task, from_checkpoint=True)
        model_config["vocab_size"] = len(task.source_dictionary)
        tokenizer = ByteLevelBPETokenizer(
            f"{args.vocab_dir}/{args.vocab_name}-vocab.json",
            f"{args.vocab_dir}/{args.vocab_name}-merges.txt",
            add_prefix_space=True,
        )
        model_config["unk_token_id"] = tokenizer.token_to_id(UNK_TOKEN)
        model_config["bos_token_id"] = tokenizer.token_to_id(BOS_TOKEN)
        model_config["eos_token_id"] = tokenizer.token_to_id(EOS_TOKEN)
        model_config["pad_token_id"] = tokenizer.token_to_id(PAD_TOKEN)

    configuration = Data2Vec2MultiConfig(**model_config)
    print(f"*** HuggingFace configuration ***\n{configuration}")

    hf_model = Data2Vec2MultiModel(configuration)
    keys = list(state_dict.keys())
    for k in keys:
        if "ema" in k or "decoder" in k:
            del state_dict[k]
    hf_model.load_state_dict(state_dict, strict=True)
    hf_model.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)

    test_model = Data2Vec2MultiModel.from_pretrained(pytorch_dump_folder_path)
    test_model.eval()
    test_model.freeze_feature_encoder()
    for n, p in test_model.named_parameters():
        compare_tensors(p, state_dict[n])


@torch.no_grad()
def test_converted_weights(args):
    checkpoint_path = args.checkpoint_path
    pytorch_dump_folder_path = args.pytorch_dump_folder_path

    # HuggingFace model
    hf_model = Data2Vec2MultiModel.from_pretrained(pytorch_dump_folder_path)
    print(f"Pre-trained weights loaded to HF model!")
    hf_model.eval()

    # checking AUDIO model
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

    if args.vocab_dir is not None:
        configuration = hf_model.config
        print(f"Comparing outputs with randomized tensors...")
        input_values = torch.randint(configuration.vocab_size - 1, (1, configuration.modalities["text"]["max_source_positions"]), dtype=torch.int64)
        print(f"Forward using HF model...")
        hf_output = hf_model(input_values, padding_mask=None, mode="TEXT", mask=False)
        print(f"Forward using fairseq model...")
        fairseq_output = fairseq_model(source=input_values, padding_mask=None, mask=False, features_only=True)
        print(f"Comparing outputs...")
        compare_tensors(hf_output.last_hidden_state, fairseq_output["x"])
        print(f'MATCHED!')
        print("*"*100)

        print(f"Comparing outputs for SAMPLE TEXT: {SAMPLE_TEXT}")
        tokenizer = ByteLevelBPETokenizer(
            f"{args.vocab_dir}/{args.vocab_name}-vocab.json",
            f"{args.vocab_dir}/{args.vocab_name}-merges.txt",
            add_prefix_space=True,
        )
        encoded = tokenizer.encode(SAMPLE_TEXT)
        encoded_ids = [BOS_TOKEN_ID] + encoded.ids # need to prepend BOS token <s>
        print(f"encoded.ids: {encoded.ids}")
        input_values = torch.tensor(
            encoded_ids, dtype=torch.int64
        ).unsqueeze(0)
        print(f"Forward using HF model...")
        hf_output = hf_model(input_values, padding_mask=None, mode="TEXT", mask=False)
        print(f"Forward using fairseq model...")
        fairseq_output = fairseq_model(source=input_values, padding_mask=None, mask=False, features_only=True)
        print(f"Comparing outputs...")
        compare_tensors(hf_output.last_hidden_state, fairseq_output["x"])
        print(f'MATCHED!')
        print("*"*100)
        
    else:
        # # compare keys and parameters
        # hf_keys = [n for n, _ in hf_model.named_parameters()]
        # fairseq_keys = [n for n, _ in fairseq_model.named_parameters()]
        # diffs = list(set(fairseq_keys) - set(hf_keys))
        # if len(diffs) > 0:
        #     print(f"diffs: {diffs}")
        # for n, p in hf_model.named_parameters():
        #     compare_tensors(p, fairseq_model.state_dict()[n])

        print(f"Comparing outputs with randomized tensors...")
        input_values = torch.randn((3, 320000), dtype=torch.float32)
        print(f"Forward using HF model...")
        hf_output = hf_model(input_values, padding_mask=None, mode="AUDIO", mask=False)
        print(f"Forward using fairseq model...")
        fairseq_output = fairseq_model(source=input_values, padding_mask=None, mask=False, features_only=True)
        print(f"Comparing outputs...")
        compare_tensors(hf_output.last_hidden_state, fairseq_output["x"])
        print(f'MATCHED!')
        print("*"*100)

        print(f"Comparing outputs from dummy datasets...")
        print("Loading processor...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")
        print("Loading dataset...")
        mls = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", cache_dir="downloaded_data", trust_remote_code=True)
        input_audio = [x["array"] for x in mls[:4]["audio"]]
        inputs = processor(input_audio, return_tensors="pt", padding=True)
        input_values = inputs.input_values
        attention_mask = inputs.attention_mask

        print(f"Forward using HF model...")
        hf_output = hf_model(input_values, padding_mask=(1-attention_mask), mode="AUDIO", mask=False)
        print(f"Forward using fairseq model...")
        fairseq_output = fairseq_model(source=input_values, padding_mask=(1-attention_mask), mask=False, features_only=True)
        print(f"Comparing outputs...")
        compare_tensors(hf_output.last_hidden_state, fairseq_output["x"])
        print(f'MATCHED!')
        print("*"*100)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--user-dir", default="examples/data2vec")
    parser.add_argument("--vocab-dir", default=None)
    parser.add_argument("--vocab-name", default="bpe-bytelevel", type=str)
    parser.add_argument("--do_convert", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    args = parser.parse_args()

    if args.do_convert:
        convert_data2vec2_checkpoint(args)
    if args.do_test:
        test_converted_weights(args)