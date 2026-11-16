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

model = StyleTTS2.from_original("path/to/checkpoint_dir/")
output = model.synthesize("Hello, how are you?")
wav = output["wav"]
```

## Kokoro

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) is a lightweight (82M
parameters) multilingual model by hexgrad based on StyleTTS 2, but without the
style encoders and style diffusion: each speaker is represented by a
precomputed "voice", a set of style vectors. Voice cloning is therefore not
supported. As for StyleTTS 2, Coqui supports inference with the original
checkpoint, converted on the fly when loaded.

| Model name | Languages | Voices |
|---|---|---|
| `tts_models/multilingual/multi-dataset/kokoro` | `en-us`, `en-gb`, `es`, `fr-fr`, `hi`, `it`, `pt-br` | 54 |

The model produces 24 kHz audio. The [available
voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) can be
listed with `tts.speakers`. The first letter of a voice name indicates its
language (`a`: American English, `b`: British English, `e`: Spanish, `f`:
French, `h`: Hindi, `i`: Italian, `p`: Brazilian Portuguese) and the second
letter its gender. Japanese (`j`) and Mandarin Chinese (`z`) voices are
included, but these languages are not supported yet.

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/kokoro")
tts.tts_to_file(text="Hello, how are you?", speaker="af_heart", language="en-us", file_path="output.wav")

# Mix voices by separating them with commas and change the speaking rate
tts.tts_to_file(text="Hola, ¿qué tal?", speaker="ef_dora,af_heart", language="es", speed=1.2, file_path="output.wav")
```

```console
tts --model_name tts_models/multilingual/multi-dataset/kokoro \
    --text "This is an example." \
    --speaker_idx af_heart \
    --language_idx en-us \
    --out_path output.wav
```

Other checkpoints in the original Kokoro format can be loaded with
`Kokoro.from_original("path/to/checkpoint_dir/")` from a directory containing
the original `config.json`, the `.pth` checkpoint and a `voices` folder with
the `.pt` voice files.

### Phonemization

Kokoro was trained on phonemes from [misaki](https://github.com/hexgrad/misaki).
For all supported languages except English, misaki uses
[espeak-ng](https://github.com/espeak-ng/espeak-ng) and Coqui produces the same
phonemes without needing misaki. For English, misaki mainly relies on its own
pronunciation dictionaries. Coqui uses espeak-ng for English by default, which
can lead to some pronunciations differing from what the model was trained on.
If misaki is installed in your environment (`pip install "misaki[en]"`), it is
automatically used for English instead.

## Important resources & papers
- Original repository: https://github.com/yl4579/StyleTTS2
- Kokoro repository: https://github.com/hexgrad/kokoro
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

## Kokoro Model
```{eval-rst}
.. autoclass:: TTS.tts.models.kokoro.Kokoro
    :members:
```
