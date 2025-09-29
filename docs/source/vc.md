# Voice conversion

## Overview

Voice conversion (VC) converts the voice in a speech signal from one speaker to
that of another speaker while preserving the linguistic content. Coqui supports
both voice conversion on its own, as well as applying it after speech synthesis
to enable multi-speaker output with single-speaker TTS models.

### Python API

Converting the voice in `source_wav` to the voice of `target_wav` (the latter
can also be a list of files):

```python
from TTS.api import TTS

tts = TTS("voice_conversion_models/multilingual/vctk/freevc24").to("cuda")
tts.voice_conversion_to_file(
  source_wav="my/source.wav",
  target_wav="my/target.wav",
  file_path="output.wav"
)
```

You can cache the target voice by setting a custom ID in `speaker` and reusing
that later instead of passing the reference audio again.
```python
from TTS.api import TTS

tts = TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2").to("cuda")

# First call with target audio
tts.voice_conversion_to_file(
  source_wav="my/source.wav",
  target_wav="my/target.wav",
  speaker="MySpeaker"
  file_path="output.wav"
)

# Later calls don't need target audio
tts.voice_conversion_to_file(
  source_wav="my/source.wav",
  speaker="MySpeaker"
  file_path="output.wav"
)
```

Voice cloning by combining TTS and VC. The FreeVC model is used for voice
conversion after synthesizing speech.

```python

tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
tts.tts_with_vc_to_file(
  "Wie sage ich auf Italienisch, dass ich dich liebe?",
  speaker_wav=["target1.wav", "target2.wav"],
  file_path="output.wav"
)
```

Some models, including [XTTS](models/xtts.md), support voice cloning directly
and a separate voice conversion step is not necessary:

```python
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
wav = tts.tts(
  text="Hello world!",
  speaker_wav="my/cloning/audio.wav",
  language="en"
)
```

### CLI

```sh
tts --model_name "voice_conversion_models/multilingual/multi-dataset/knnvc" \
    --source_wav "source.wav" \
    --target_wav "target1.wav" "target2.wav" \
    --out_path "output.wav"
```

You can also cache the target speaker by assigning a custom ID in
`--speaker_idx`, so that target audio is not required for subsequent calls.

## Pretrained models

Coqui includes the following pretrained voice conversion models. Training is not
supported.

### FreeVC

- `voice_conversion_models/multilingual/vctk/freevc24`

Adapted from: https://github.com/OlaWod/FreeVC

### kNN-VC

- `voice_conversion_models/multilingual/multi-dataset/knnvc`

At least 1-5 minutes of target speaker data are recommended.

Adapted from: https://github.com/bshall/knn-vc

### OpenVoice

- `voice_conversion_models/multilingual/multi-dataset/openvoice_v1`
- `voice_conversion_models/multilingual/multi-dataset/openvoice_v2`

Adapted from: https://github.com/myshell-ai/OpenVoice
