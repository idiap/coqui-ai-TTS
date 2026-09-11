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

# Let OmniVoice select a voice automatically.
tts.tts_to_file("Hello from OmniVoice.", language="en", file_path="output.wav")

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

OmniVoice's code is Apache-2.0 licensed. Its pretrained weights are CC-BY-NC;
review those noncommercial terms before downloading or using the catalogue model.


```{toctree}
:hidden:
cloning
vc
server
marytts
```
