# Pantagruel

This folder contains codebase for loading and using the speech-only and text-only pre-trained models, which are based on [data2vec 2.0 architecture](https://arxiv.org/abs/2212.07525). The pre-trained models were trained using `fairseq` v1 library.

## Current models
Current available pre-trained models are saved under the following directory `/lustre/fsstor/projects/rech/oou/commun/pretrained_models`, including
- `Speech_Base_fr_1K`: trained on around 1K hours of the French subset of Multilingual LibriSpeech corpus 
- `Speech_Large_fr_14K`: trained on 14K hours of LeBenchmark dataset
- `Speech_Large_fr_14K_v1`: same as `Speech_Large_fr_14K` but used a different training settings for more epochs. Preliminary experiments on CommonVoice ASR show better performance than `Speech_Large_fr_14K`
- `camembert-base-wikipedia-4gb`: trained on similar pre-training corpus as `Text_Base_fr_4GB_v0`.
- `Text_Base_fr_4GB_v0`: trained on around 4GB of text from French Wikipedia 2019 dump, tokenizer was learned on the same pre-training data
- `Text_Base_fr_4GB_v1`: trained on the same data as `Text_Base_fr_4GB_v0`, but tokenizer was learned on the subset french-30b of croissantLLM dataset. 

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
    RobertaTokenizerFast,
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

# Forward pass
hf_output = hf_model(input_values=normalized_input_values)
extracted_features = hf_output.last_hidden_state

# TEXT-ONLY MODEL
hf_model = Data2Vec2MultiModel.from_pretrained(text_model_dir)
hf_model.eval()
hf_model.freeze_feature_encoder()

# Tokenize text
SAMPLE_TEXT = "Bonjour le monde !!"
## For Text_Base_fr_4GB_v0
tokenizer = RobertaTokenizerFast.from_pretrained(
           text_model_dir.as_posix(), add_prefix_space=False, unicode_normalizer="nfc"
        )
encoded_ids = tokenizer(SAMPLE_TEXT)["input_ids"]

## For Text_Base_fr_4GB_v1
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(text_model_dir / "tokenizer_fast")
encoded_ids = tokenizer.encode(SAMPLE_TEXT)

input_ids = torch.tensor(encoded_ids, dtype=torch.int64).unsqueeze(0)

# Forward pass
hf_output = hf_model(input_ids=input_ids)
extracted_features = hf_output.last_hidden_state
```
