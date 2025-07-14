# 🐶 Bark

Bark is a multi-lingual TTS model created by [Suno-AI](https://www.suno.ai/). It can generate conversational speech as well as  music and sound effects.
It is architecturally very similar to Google's [AudioLM](https://arxiv.org/abs/2209.03143). For more information, please refer to the [Suno-AI's repo](https://github.com/suno-ai/bark).


## Acknowledgements
- 👑[Suno-AI](https://www.suno.ai/) for training and open-sourcing this model.
- 👑[gitmylo](https://github.com/gitmylo) for finding [the solution](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/) to the semantic token generation for voice clones and finetunes.
- 👑[serp-ai](https://github.com/serp-ai/bark-with-voice-clone) for controlled voice cloning.


## Example use

```{seealso}
[Voice cloning](../cloning.md)
```

```python
text = "Hello, my name is Manmay , how are you?"

from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark

config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="path/to/model/dir/", eval=True)

# Random speaker
output_dict = model.synthesize(text)

# Cloning a speaker.
output_dict = model.synthesize(text, speaker_wav="path/to/speaker.wav")
```

Using 🐸TTS API:

```python
from TTS.api import TTS

# Load the model to GPU
# Bark is really slow on CPU, so we recommend using GPU.
tts = TTS("tts_models/multilingual/multi-dataset/bark").to("cuda")


# Clone voice and cache it with the custom ID `ljspeech`.
tts.tts_to_file(text="Hello, my name is Manmay , how are you?",
                file_path="output.wav",
                speaker_wav=["tests/data/ljspeech/wavs/LJ001-0001.wav"],
                speaker="ljspeech")


# When you run it again it uses the stored values to generate the voice.
tts.tts_to_file(text="Hello, my name is Manmay , how are you?",
                file_path="output.wav",
                speaker="ljspeech")


# random speaker
tts = TTS("tts_models/multilingual/multi-dataset/bark").to("cuda")
tts.tts_to_file("hello world", file_path="out.wav")
```

Using 🐸TTS Command line:

```console
# Clone the `ljspeech` voice and cache it under that ID for later reuse without reference audio.
tts --model_name  tts_models/multilingual/multi-dataset/bark \
    --text "This is an example." \
    --out_path "output.wav" \
    --speaker_wav tests/data/ljspeech/wavs/*.wav
    --speaker_idx "ljspeech"

# Random voice generation
tts --model_name  tts_models/multilingual/multi-dataset/bark \
    --text "This is an example." \
    --out_path "output.wav"
```

## Important resources & papers
- Original Repo: https://github.com/suno-ai/bark
- Cloning implementation: https://github.com/serp-ai/bark-with-voice-clone
- AudioLM: https://arxiv.org/abs/2209.03143

## BarkConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.bark_config.BarkConfig
    :members:
```

## Bark Model
```{eval-rst}
.. autoclass:: TTS.tts.models.bark.Bark
    :members:
```
