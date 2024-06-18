# Pantagruel

This folder contains codebase for loading and using the speech-only and text-only pre-trained models, which are based on [data2vec 2.0 architecture](https://arxiv.org/abs/2212.07525). The pre-trained models were trained using `fairseq` v1 library.

## Current models
Current available pre-trained models are saved under the follwing directory: `/lus/work/CT10/lig3801/SHARED/pretrained_models`, including:
- `Speech_Base_fr_1K`: trained on around 1K hours of the French subset of Multilingual LibriSpeech corpus 
- `Text_Base_fr_4GB_v0`: trained on around 5GB of text from French Wikipedia 2019 dump
- `camembert-base-wikipedia-4gb`: trained on similar pre-training corpus as `Text_Base_fr_4GB_v0`.

The converted HuggingFace models are saved under sub-folder named `HuggingFace` in corresponding model-specific folders.


## Feature extraction
To extract representations for a given audio or textual input, the pre-trained speech-only or text-only models can be used as follows:
```python
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    Data2Vec2MultiConfig,
    Data2Vec2MultiModel,
)
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

pretrained_dir = Path("/lus/work/CT10/lig3801/SHARED/pretrained_models")
audio_model_dir = pretrained_dir / "Speech_Base_fr_1K" / "HuggingFace"
text_model_dir = pretrained_dir / "Text_Base_fr_4GB_v0" / "HuggingFace"

# SPEECH-ONLY MODEL
hf_model = Data2Vec2MultiModel.from_pretrained(audio_model_dir)
hf_model.eval()
hf_model.freeze_feature_encoder()

# Important: normalized audio input signal
input_values = torch.randn((3, 320000), dtype=torch.float32)
with torch.no_grad():
    normalized_input_values = F.layer_norm(input_values, input_values.size()[1:])

hf_output = hf_model(normalized_input_values, mode="AUDIO")
extracted_features = hf_output.last_hidden_state

# TEXT-ONLY MODEL
hf_model = Data2Vec2MultiModel.from_pretrained(text_model_dir)
hf_model.eval()
hf_model.freeze_feature_encoder()

SAMPLE_TEXT = "Bonjour le monde !!"

tokenizer = ByteLevelBPETokenizer(
    (text_model_dir / "bpe-bytelevel-vocab.json").as_posix(),
    (text_model_dir / "bpe-bytelevel-merges.txt").as_posix(),
    add_prefix_space=False,
    unicode_normalizer="nfc")
tokenizer.add_special_tokens(SPECIAL_TOKENS)

encoded = tokenizer.encode(SAMPLE_TEXT)
# prepend BOS token <s>
encoded_ids = [tokenizer.token_to_id(BOS_TOKEN)] + encoded.ids
input_values = torch.tensor(encoded_ids, dtype=torch.int64).unsqueeze(0)
hf_output = hf_model(input_values, mode="TEXT")
extracted_features = hf_output.last_hidden_state
```
