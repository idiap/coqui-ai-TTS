(synthesizing_speech)=
# Synthesizing speech

## Overview

Coqui TTS provides three main methods for inference:

1. 🐍Python API
2. TTS command line interface (CLI)
3. [Local demo server](server.md)

```{include} ../../README.md
:start-after: <!-- start inference -->
```

## OmniVoice

Install the optional integration and load OmniVoice through the regular Coqui API:

```bash
pip install "coqui-tts[omnivoice]"
```

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/omnivoice").to("cuda")

# OmniVoice reports its complete language list at runtime (currently 646 IDs).
print(len(tts.languages))
print(tts.languages)

# Let OmniVoice select a voice automatically in different languages.
tts.tts_to_file("Hello from OmniVoice.", language="en", file_path="output.wav")
tts.tts_to_file("Xin chào từ OmniVoice.", language="vi", file_path="output-vi.wav")
tts.tts_to_file("Hola desde OmniVoice.", language="es", file_path="output-es.wav")

# Clone a voice. Supplying the transcript avoids loading an ASR model.
tts.tts_to_file(
    "This uses a cloned voice.",
    language="en",
    speaker_wav="reference.wav",
    ref_text="Transcript of the reference audio.",
    file_path="cloned.wav",
)

# Design a voice with a natural-language instruction.
tts.tts_to_file(
    "This uses a designed voice.",
    language="en",
    instruct="male, british accent",
    file_path="designed.wav",
)
```

The `language` argument selects one of the language IDs returned by
`tts.languages`; `en` is used for the cloning and voice-design examples only
because their example text is English. The integration does not hard-code an
English-only language list.

OmniVoice's code is Apache-2.0 licensed. Its pretrained weights are CC-BY-NC;
review those noncommercial terms before downloading or using the catalogue model.


```{toctree}
:hidden:
cloning
vc
server
marytts
```
