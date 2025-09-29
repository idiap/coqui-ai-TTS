# Project structure

## Directory structure

A non-comprehensive overview of the Coqui source code:

| Directory | Contents |
| - | - |
| **Core** | |
| **[`TTS/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS)** | Main source code |
| **[`-   .models.json`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/.models.json)** | Pretrained model list |
| **[`-   api.py`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/api.py)** | Python API |
| **[`-   bin/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/bin)** | Executables and CLI |
| **[`-   tts/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/tts)** | Text-to-speech models |
| **[`-       configs/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/tts/configs)** | Model configurations |
| **[`-       layers/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/tts/layers)** | Model layer definitions |
| **[`-       models/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/tts/models)** | Model definitions |
| **[`-   vc/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/vc)** | Voice conversion models |
| `-       (same)` | |
| **[`-   vocoder/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/vocoder)** | Vocoder models |
| `-       (same)` | |
| **[`-   encoder/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/TTS/encoder)** | Speaker encoder models |
| `-       (same)` | |
| **Recipes/notebooks** | |
| **[`notebooks/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/notebooks)** | Jupyter Notebooks for model evaluation, parameter selection and data analysis |
| **[`recipes/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/recipes)** | Training recipes |
| **Others** | |
| **[`pyproject.toml`](https://github.com/idiap/coqui-ai-TTS/tree/dev/pyproject.toml)** | Project metadata, configuration and dependencies |
| **[`docs/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/docs)** | Documentation |
| **[`tests/`](https://github.com/idiap/coqui-ai-TTS/tree/dev/tests)** | Unit and integration tests |
