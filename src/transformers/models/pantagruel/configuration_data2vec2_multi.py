# coding=utf-8
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# 
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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


""" Data2Vec2 multi configuration"""

import os
from typing import Union, Dict, Any, Optional
from ...dynamic_module_utils import custom_object_save
from ...utils import CONFIG_NAME
from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MyPretrainedConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def to_json_string(self, use_diff: bool = False) -> str:
        return super().to_json_string(use_diff)

    def update(self, config_dict):
        for key, value in config_dict.items():
            if not hasattr(self, key):
                continue
            if isinstance(getattr(self, key), MyPretrainedConfig):
                getattr(self, key).update(config_dict[key])
            else:
                setattr(self, key, value)

    # Copied from the parent class, only changed use_diff from True to False to correctly save nested config class
    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self._set_token_in_kwargs(kwargs)

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        non_default_generation_parameters = {}
        for parameter_name, default_value in self._get_generation_defaults().items():
            if hasattr(self, parameter_name) and getattr(self, parameter_name) != default_value:
                non_default_generation_parameters[parameter_name] = getattr(self, parameter_name)
        if len(non_default_generation_parameters) > 0:
            logger.warning(
                "Some non-default generation parameters are set in the model config. These should go into a "
                "GenerationConfig file (https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model) "
                "instead. This warning will be raised to an exception in v4.41.\n"
                f"Non-default generation parameters: {str(non_default_generation_parameters)}"
            )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=False)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    # Copied from the parent class, change the instantiation and updating of class from config_dict to correctly load nested config
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "MyPretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PretrainedConfig.get_config_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from those parameters.
        """
        config_tuple = super().from_dict(config_dict, **kwargs)
        print(config_tuple)
        if config_tuple is tuple:
            config = config_tuple[0]
            kwargs = config_tuple[1]
        else:
            config = config_tuple
            kwargs = None
        # My updated config
        for key, value in config_dict.items():
            if not hasattr(config, key):
                continue
            if isinstance(getattr(config, key), MyPretrainedConfig):
                getattr(config, key).update(config_dict[key])
        if kwargs is not None:
            return config, kwargs
        else:
            return config


class D2v2ModalityConfig(MyPretrainedConfig):
    def __init__(
        self,
        type="AUDIO",
        prenet_depth=4,
        prenet_layerdrop=0,
        prenet_dropout=0.0,
        start_drop_path_rate=0.0,
        end_drop_path_rate=0.0,
        num_extra_tokens=0,
        init_extra_token_zero=True,
        mask_noise_std=0.01,
        mask_prob_min=None,
        mask_prob=0.7,
        inverse_mask=False,
        mask_prob_adjust=0.0,
        keep_masked_pct=0.0,
        mask_length=5,
        add_masks=False,
        remove_masks=False,
        mask_dropout=0.0,
        encoder_zero_mask=True,
        mask_channel_prob=0.0,
        mask_channel_length=64,
        local_grad_mult=1.0,
        use_alibi_encoder=False,
        alibi_scale=1.0,
        learned_alibi=False,
        alibi_max_pos=None,
        learned_alibi_scale=False,
        learned_alibi_scale_per_head=False,
        learned_alibi_scale_per_layer=False,
        num_alibi_heads=12,
        model_depth=12,
        ema_local_encoder=False,
        decoder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = type
        self.prenet_depth = prenet_depth
        self.prenet_layerdrop = prenet_layerdrop
        self.prenet_dropout = prenet_dropout
        self.start_drop_path_rate = start_drop_path_rate
        self.end_drop_path_rate = end_drop_path_rate
        self.num_extra_tokens = num_extra_tokens
        self.init_extra_token_zero = init_extra_token_zero
        self.mask_noise_std = mask_noise_std
        self.mask_prob_min = mask_prob_min
        self.mask_prob = mask_prob
        self.inverse_mask = inverse_mask
        self.mask_prob_adjust = mask_prob_adjust
        self.keep_masked_pct = keep_masked_pct
        self.mask_length = mask_length
        self.add_masks = add_masks
        self.remove_masks = remove_masks
        self.mask_dropout = mask_dropout
        self.encoder_zero_mask = encoder_zero_mask
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length
        self.local_grad_mult = local_grad_mult
        self.use_alibi_encoder = use_alibi_encoder
        self.alibi_scale = alibi_scale
        self.learned_alibi = learned_alibi
        self.alibi_max_pos = alibi_max_pos
        self.learned_alibi_scale = learned_alibi_scale
        self.learned_alibi_scale_per_head = learned_alibi_scale_per_head
        self.learned_alibi_scale_per_layer = learned_alibi_scale_per_layer
        self.num_alibi_heads = num_alibi_heads
        self.model_depth = model_depth


class D2v2AudioConfig(D2v2ModalityConfig):
    """
    Configuration including common args and args specific to audio-only pre-training
    """
    def __init__(
        self, 
        extractor_mode="layer_norm",
        feature_encoder_spec="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        conv_pos_width=95,
        conv_pos_groups=16,
        conv_pos_depth=5,
        conv_pos_pre_ln=False,
        **kwargs,
    ):
        super().__init__(type="AUDIO", **kwargs)
        self.extractor_mode = extractor_mode
        self.feature_encoder_spec = feature_encoder_spec
        self.conv_pos_width = conv_pos_width
        self.conv_pos_groups = conv_pos_groups
        self.conv_pos_depth = conv_pos_depth
        self.conv_pos_pre_ln = conv_pos_pre_ln


class D2v2TextConfig(D2v2ModalityConfig):
    """
    Configuration including common args and args specific to text-only pre-training
    """
    def __init__(
        self,
        vocab_size=50000,
        unk_token_id=3,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        max_source_positions=512,
        learned_pos=True,
        dropout=0.1,
        no_scale_embedding=True,
        layernorm_embedding=True,
        no_token_positional_embeddings=False,
        **kwargs,
    ):
        super().__init__(type="TEXT", **kwargs)
        self.vocab_size = vocab_size
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_source_positions = max_source_positions
        self.learned_pos = learned_pos
        self.dropout = dropout
        self.no_scale_embedding = no_scale_embedding
        self.layernorm_embedding = layernorm_embedding
        self.no_token_positional_embeddings = no_token_positional_embeddings


class D2v2ModalitiesConfig(MyPretrainedConfig):
    def __init__(
        self, 
        audio_config=D2v2AudioConfig(),
        text_config=D2v2TextConfig(),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio = audio_config
        self.text = text_config


class Data2Vec2MultiConfig(MyPretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Data2Vec2MultiModel`]. It is used to instantiate
    an Data2Vec2MultiModel model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        depth (`int`, *optional*, defaults to 12):
            Number of Transformer layers in the encoder.

    Example:

    ```python
    >>> from transformers import Data2Vec2MultiConfig, Data2Vec2MultiModel

    >>> # Initializing a Data2Vec2MultiConfig for audio
    >>> configuration = Data2Vec2MultiConfig()

    >>> # Initializing a model (with random weights) with the configuration
    >>> model = Data2Vec2MultiModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "data2vec2-multi"

    def __init__(
        self,
        depth=12,
        start_drop_path_rate=0.0,
        end_drop_path_rate=0.0,
        num_heads=12,
        norm_eps=1e-5,
        norm_affine=True,
        encoder_dropout=0.1,
        post_mlp_drop=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        dropout_input=0.0,
        layerdrop=0.0,
        embed_dim=768,
        mlp_ratio=4.0,
        layer_norm_first=False,
        end_of_block_targets=False,
        clone_batch=1,
        log_norms=True,
        modalities=D2v2ModalitiesConfig(),
        supported_modality="AUDIO",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.start_drop_path_rate = start_drop_path_rate
        self.end_drop_path_rate = end_drop_path_rate

        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.norm_affine = norm_affine
        self.post_mlp_drop = post_mlp_drop
        self.encoder_dropout = encoder_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout_input = dropout_input
        self.layerdrop = layerdrop
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.layer_norm_first = layer_norm_first
        self.end_of_block_targets = end_of_block_targets
        self.clone_batch = clone_batch
        self.log_norms = log_norms

        if isinstance(modalities, D2v2ModalitiesConfig):
            self.modalities = modalities
        else:
            self.modalities = D2v2ModalitiesConfig()
        self.supported_modality = supported_modality

        # Attributes for hopsparser
        self.hidden_size = embed_dim
        self.num_layers = depth
        self.n_layers = depth
        self.num_hidden_layers = depth