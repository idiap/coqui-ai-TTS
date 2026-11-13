# StyleTTS 2

StyleTTS 2 is an English TTS model by [Yinghao Aaron Li et
al.](https://arxiv.org/abs/2306.07691) that models speaking style as a latent
random variable sampled with a diffusion model, so it can produce a style that
suits the text without needing reference speech. It is trained adversarially
with a large speech language model (WavLM) as discriminator and uses
differentiable duration modelling. The multi-speaker model additionally
supports zero-shot voice cloning from a short reference audio.

Coqui currently supports inference with the original pre-trained checkpoints,
which are converted to the Coqui format on the fly when loaded. Training is not
supported yet.

```{seealso}
[Voice cloning](../cloning.md)
```

## Pre-trained models

| Model name | Dataset | Voice cloning |
|---|---|---|
| `tts_models/en/ljspeech/styletts2` | LJSpeech (single speaker) | No |
| `tts_models/en/libritts/styletts2` | LibriTTS (multi-speaker) | Yes, a reference voice is required |

Both models produce 24 kHz audio. The checkpoints are downloaded from the
authors' Hugging Face repositories
([LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech),
[LibriTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS)).

```{important}
The authors' terms for the pre-trained models: you must inform listeners that
the speech samples are synthesized by the pre-trained models, unless you have
permission to use the voice you synthesize. See the [original
repository](https://github.com/yl4579/StyleTTS2#license) for details.
```

## Example use

Using 🐸TTS API:

```python
from TTS.api import TTS

# Single speaker
tts = TTS("tts_models/en/ljspeech/styletts2")
tts.tts_to_file(text="Hello, how are you?", file_path="output.wav")

# Voice cloning, caching the voice with the custom ID `ljspeech`
tts = TTS("tts_models/en/libritts/styletts2")
tts.tts_to_file(
    text="Hello, how are you?",
    file_path="output.wav",
    speaker_wav="tests/data/ljspeech/wavs/LJ001-0001.wav",
    speaker="ljspeech",
)

# Reuse the cached voice without the reference audio
tts.tts_to_file(text="Hello, how are you?", file_path="output.wav", speaker="ljspeech")
```

When cloning a voice, the style sampled by the diffusion model is mixed with
the style of the reference audio. `alpha` (default `0.3`) controls the
acoustic style (timbre) and `beta` (default `0.7`) the prosodic style; values
closer to 0 follow the reference audio more closely:

```python
tts.tts_to_file(
    text="Hello, how are you?",
    file_path="output.wav",
    speaker_wav="tests/data/ljspeech/wavs/LJ001-0001.wav",
    alpha=0.1,
    beta=0.5,
)
```

Using 🐸TTS Command line:

```console
# Single speaker
tts --model_name tts_models/en/ljspeech/styletts2 \
    --text "This is an example." \
    --out_path output.wav

# Voice cloning
tts --model_name tts_models/en/libritts/styletts2 \
    --text "This is an example." \
    --speaker_wav tests/data/ljspeech/wavs/LJ001-0001.wav \
    --out_path output.wav
```

### Loading original checkpoints

Other checkpoints in the original StyleTTS 2 format, e.g. fine-tuned on your
own data, can be loaded from a directory containing the original `config.yml`
and `.pth` files:

```python
from TTS.tts.models.styletts2 import StyleTTS2

model, config = StyleTTS2.load_from_original_checkpoint("path/to/checkpoint_dir/")
model.eval()
output = model.synthesize("Hello, how are you?")
wav = output["wav"]
```

## Important resources & papers
- Original repository: https://github.com/yl4579/StyleTTS2
- StyleTTS 2 paper: https://arxiv.org/abs/2306.07691
- PL-BERT: https://arxiv.org/abs/2301.08810
- iSTFTNet: https://arxiv.org/abs/2203.02395
- HiFiGAN: https://arxiv.org/abs/2010.05646

## StyleTTS2Config
```{eval-rst}
.. autoclass:: TTS.tts.configs.styletts2_config.StyleTTS2Config
    :members:
```

## StyleTTS2Args
```{eval-rst}
.. autoclass:: TTS.tts.configs.styletts2_config.StyleTTS2Args
    :members:
```

## StyleTTS2 Model
```{eval-rst}
.. autoclass:: TTS.tts.models.styletts2.StyleTTS2
    :members:
```
