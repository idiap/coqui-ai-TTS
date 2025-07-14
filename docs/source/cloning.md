# Voice cloning

Some TTS models can only synthesize speech for a fixed set of voices they were
trained on. Others also support _voice cloning_, i.e., they can generate a new
voice on-the-fly from a provided reference audio file. These Coqui TTS models
allow voice cloning (the model config will contain `supports_cloning=True`):

- [YourTTS](models/vits.md) (and other d-vector based models)
- [XTTS](models/xtts.md)
- [Tortoise](models/tortoise.md)
- [Bark](models/bark.md)

All [voice conversion models](vc.md) also perform voice cloning, but with speech as the
input instead of text.

```{important}
Voice cloning raises several ethical concerns and must not be used to
impersonate individuals without their consent, deceive others, or spread
misinformation ([deepfakes](https://en.wikipedia.org/wiki/Deepfake)). We
strongly encourage you to respect the privacy and rights of individuals, and to
always ensure that your use of Coqui TTS is transparent, responsible, and
respectful of others.
```

## Usage

```{versionchanged} 0.27.0
Coqui can now cache cloned voices for easy reuse. Implementation details of this
may change in future versions.
```

Reference audio for voice cloning is passed via the `speaker_wav` argument,
which may be a single file or a list of files. If a custom speaker ID is also
passed in `speaker`, the resulting voice will be cached in `voice_dir`. If that
voice already exists, it is overwritten. Subsequent calls can then use that
`speaker` without having to provide reference audio again.

`voice_dir` defaults to a subfolder `voices/` in the folder of the model
checkpoint. For models used by name, e.g.
`tts_models/multilingual/multi-dataset/xtts_v2`, this would be
`~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/voices/`
(on Linux, see the [FAQ](faq.md#where-does-coqui-store-downloaded-models) for
default model locations on other platforms).

### Python API

```python
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a TTS model with voice cloning support
api = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 1. Clone the voice from `speaker_wav` and cache it under a custom speaker ID
api.tts_to_file(
  text="Hello world",
  speaker_wav=["my/cloning/audio.wav", "my/cloning/audio2.wav"],
  speaker="MySpeaker1",
  language="en",
)

# 2. The voice can now be reused without providing reference audio
api.tts_to_file(
  text="Hello world",
  speaker="MySpeaker1",
  language="en",
)
```

### Command-line interface

```bash
# 1. Clone the voice from `speaker_wav` and cache it under a custom speaker ID
tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2" \
    --text "Hello world" \
    --language_idx "en" \
    --speaker_wav "my/cloning/audio.wav" "my/cloning/audio2.wav" \
    --speaker_idx "MySpeaker1"

# 2. The voice can now be reused without providing reference audio
tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2" \
    --text "Hello world" \
    --language_idx "en" \
    --speaker_idx "MySpeaker1"
```

### Metadata

Some metadata is stored within the voice files to check what models they are
compatible with (Coqui does not enforce any checks yet). Let's create a voice
with the LJSpeech files stored in the repository:

```bash
tts --model_name "voice_conversion_models/multilingual/multi-dataset/knnvc" \
    --source_wav source.wav \
    --target_wav tests/data/ljspeech/wavs/*.wav \
    --speaker_idx LJ \
    --voice_dir wavlm-voices
```

We can then print the metadata:

```python
import torch
voice = torch.load("wavlm-voices/LJ.pth", map_location="cpu")
print(voice["metadata"])
```

```python
{
  'model': {'name': 'wavlm', 'layer': 6},
  'speaker_id': 'LJ',
  'source_files': [
    'tests/data/ljspeech/wavs/LJ001-0001.wav',
    'tests/data/ljspeech/wavs/LJ001-0002.wav',
    ...,
  ],
  'created_at': '2025-06-25T12:17+00:00',
  'coqui_version': '0.27.0',
}
```

## For developers

To add voice cloning support to a model, it needs to inherit from
{py:class}`~TTS.utils.voices.CloningMixin` (in addition to
{py:class}`~TTS.tts.models.base_tts.BaseTTS` or
{py:class}`~TTS.vc.models.base_vc.BaseVC`). You then only have to implement a
model-specific `_clone_voice()` method that returns speaker embeddings and
model-specific metadata. For example, for [XTTS](models/xtts.md):

```{eval-rst}
.. literalinclude:: ../../TTS/tts/models/xtts.py
    :pyobject: Xtts._clone_voice
```

Then in your model's `synthesize()` method you use the mixin's
{py:func}`~TTS.utils.voices.CloningMixin.clone_voice` to access the
voice data, while all caching is handled automatically.

### API doc

#### CloningMixin

```{eval-rst}
.. autoclass:: TTS.utils.voices.CloningMixin
    :members:
```

#### VoiceMetadata

```{eval-rst}
.. autoclass:: TTS.utils.voices.VoiceMetadata
    :members:
```
