
## 🐸Coqui TTS News
- 📣 Fork of the [original, unmaintained repository](https://github.com/coqui-ai/TTS). New PyPI package: [coqui-tts](https://pypi.org/project/coqui-tts)
- 📣 Prebuilt wheels are now also published for Mac and Windows (in addition to Linux as before) for easier installation across platforms.
- 📣 ⓍTTSv2 is here with 16 languages and better performance across the board.
- 📣 ⓍTTS fine-tuning code is out. Check the [example recipes](https://github.com/idiap/coqui-ai-TTS/tree/dev/recipes/ljspeech).
- 📣 ⓍTTS can now stream with <200ms latency.
- 📣 ⓍTTS, our production TTS model that can speak 13 languages, is released [Blog Post](https://coqui.ai/blog/tts/open_xtts), [Demo](https://huggingface.co/spaces/coqui/xtts), [Docs](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html)
- 📣 [🐶Bark](https://github.com/suno-ai/bark) is now available for inference with unconstrained voice cloning. [Docs](https://coqui-tts.readthedocs.io/en/latest/models/bark.html)
- 📣 You can use [~1100 Fairseq models](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) with 🐸TTS.

## <img src="https://raw.githubusercontent.com/idiap/coqui-ai-TTS/main/images/coqui-log-green-TTS.png" height="56"/>


**🐸TTS is a library for advanced Text-to-Speech generation.**

🚀 Pretrained models in +1100 languages.

🛠️ Tools for training new models and fine-tuning existing models in any language.

📚 Utilities for dataset analysis and curation.
______________________________________________________________________

[![Discord](https://img.shields.io/discord/1037326658807533628?color=%239B59B6&label=chat%20on%20discord)](https://discord.gg/5eXr5seRrv)
[![License](<https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg>)](https://opensource.org/licenses/MPL-2.0)
[![PyPI version](https://badge.fury.io/py/coqui-tts.svg)](https://badge.fury.io/py/coqui-tts)
[![Downloads](https://pepy.tech/badge/coqui-tts)](https://pepy.tech/project/coqui-tts)
[![DOI](https://zenodo.org/badge/265612440.svg)](https://zenodo.org/badge/latestdoi/265612440)

![GithubActions](https://github.com/idiap/coqui-ai-TTS/actions/workflows/tests.yml/badge.svg)
![GithubActions](https://github.com/idiap/coqui-ai-TTS/actions/workflows/docker.yaml/badge.svg)
![GithubActions](https://github.com/idiap/coqui-ai-TTS/actions/workflows/style_check.yml/badge.svg)
[![Docs](<https://readthedocs.org/projects/coqui-tts/badge/?version=latest&style=plastic>)](https://coqui-tts.readthedocs.io/en/latest/)

</div>

______________________________________________________________________

## 💬 Where to ask questions
Please use our dedicated channels for questions and discussion. Help is much more valuable if it's shared publicly so that more people can benefit from it.

| Type                                         | Platforms                           |
| -------------------------------------------- | ----------------------------------- |
| 🚨 **Bug Reports, Feature Requests & Ideas** | [GitHub Issue Tracker]              |
| 👩‍💻 **Usage Questions**                       | [GitHub Discussions]                |
| 🗯 **General Discussion**                    | [GitHub Discussions] or [Discord]   |

[github issue tracker]: https://github.com/idiap/coqui-ai-TTS/issues
[github discussions]: https://github.com/idiap/coqui-ai-TTS/discussions
[discord]: https://discord.gg/5eXr5seRrv
[Tutorials and Examples]: https://github.com/coqui-ai/TTS/wiki/TTS-Notebooks-and-Tutorials

The [issues](https://github.com/coqui-ai/TTS/issues) and
[discussions](https://github.com/coqui-ai/TTS/discussions) in the original
repository are also still a useful source of information.


## 🔗 Links and Resources
| Type                            | Links                               |
| ------------------------------- | --------------------------------------- |
| 💼 **Documentation**              | [ReadTheDocs](https://coqui-tts.readthedocs.io/en/latest/)
| 💾 **Installation**               | [TTS/README.md](https://github.com/idiap/coqui-ai-TTS/tree/dev#installation)|
| 👩‍💻 **Contributing**               | [CONTRIBUTING.md](https://github.com/idiap/coqui-ai-TTS/blob/main/CONTRIBUTING.md)|
| 🚀 **Released Models**            | [Standard models](https://github.com/idiap/coqui-ai-TTS/blob/dev/TTS/.models.json) and [Fairseq models in ~1100 languages](https://github.com/idiap/coqui-ai-TTS#example-text-to-speech-using-fairseq-models-in-1100-languages-)|

## Features
- High-performance Deep Learning models for Text2Speech tasks. See lists of models below.
- Fast and efficient model training.
- Detailed training logs on the terminal and Tensorboard.
- Support for Multi-speaker TTS.
- Efficient, flexible, lightweight but feature complete `Trainer API`.
- Released and ready-to-use models.
- Tools to curate Text2Speech datasets under```dataset_analysis```.
- Utilities to use and test your models.
- Modular (but not too much) code base enabling easy implementation of new ideas.

## Model Implementations
### Spectrogram models
- Tacotron: [paper](https://arxiv.org/abs/1703.10135)
- Tacotron2: [paper](https://arxiv.org/abs/1712.05884)
- Glow-TTS: [paper](https://arxiv.org/abs/2005.11129)
- Speedy-Speech: [paper](https://arxiv.org/abs/2008.03802)
- Align-TTS: [paper](https://arxiv.org/abs/2003.01950)
- FastPitch: [paper](https://arxiv.org/pdf/2006.06873.pdf)
- FastSpeech: [paper](https://arxiv.org/abs/1905.09263)
- FastSpeech2: [paper](https://arxiv.org/abs/2006.04558)
- SC-GlowTTS: [paper](https://arxiv.org/abs/2104.05557)
- Capacitron: [paper](https://arxiv.org/abs/1906.03402)
- OverFlow: [paper](https://arxiv.org/abs/2211.06892)
- Neural HMM TTS: [paper](https://arxiv.org/abs/2108.13320)
- Delightful TTS: [paper](https://arxiv.org/abs/2110.12612)

### End-to-End Models
- ⓍTTS: [blog](https://coqui.ai/blog/tts/open_xtts)
- VITS: [paper](https://arxiv.org/pdf/2106.06103)
- 🐸 YourTTS: [paper](https://arxiv.org/abs/2112.02418)
- 🐢 Tortoise: [orig. repo](https://github.com/neonbjb/tortoise-tts)
- 🐶 Bark: [orig. repo](https://github.com/suno-ai/bark)

### Attention Methods
- Guided Attention: [paper](https://arxiv.org/abs/1710.08969)
- Forward Backward Decoding: [paper](https://arxiv.org/abs/1907.09006)
- Graves Attention: [paper](https://arxiv.org/abs/1910.10288)
- Double Decoder Consistency: [blog](https://erogol.com/solving-attention-problems-of-tts-models-with-double-decoder-consistency/)
- Dynamic Convolutional Attention: [paper](https://arxiv.org/pdf/1910.10288.pdf)
- Alignment Network: [paper](https://arxiv.org/abs/2108.10447)

### Speaker Encoder
- GE2E: [paper](https://arxiv.org/abs/1710.10467)
- Angular Loss: [paper](https://arxiv.org/pdf/2003.11982.pdf)

### Vocoders
- MelGAN: [paper](https://arxiv.org/abs/1910.06711)
- MultiBandMelGAN: [paper](https://arxiv.org/abs/2005.05106)
- ParallelWaveGAN: [paper](https://arxiv.org/abs/1910.11480)
- GAN-TTS discriminators: [paper](https://arxiv.org/abs/1909.11646)
- WaveRNN: [origin](https://github.com/fatchord/WaveRNN/)
- WaveGrad: [paper](https://arxiv.org/abs/2009.00713)
- HiFiGAN: [paper](https://arxiv.org/abs/2010.05646)
- UnivNet: [paper](https://arxiv.org/abs/2106.07889)

### Voice Conversion
- FreeVC: [paper](https://arxiv.org/abs/2210.15418)

You can also help us implement more models.

## Installation
🐸TTS is tested on Ubuntu 22.04 with **python >= 3.9, < 3.13.**.

If you are only interested in [synthesizing speech](https://coqui-tts.readthedocs.io/en/latest/inference.html) with the released 🐸TTS models, installing from PyPI is the easiest option.

```bash
pip install coqui-tts
```

If you plan to code or train models, clone 🐸TTS and install it locally.

```bash
git clone https://github.com/idiap/coqui-ai-TTS
cd coqui-ai-TTS
pip install -e .
```

### Optional dependencies

The following extras allow the installation of optional dependencies:

| Name | Description |
|------|-------------|
| `all` | All optional dependencies, except `dev` and `docs` |
| `dev` | Development dependencies |
| `docs` | Dependencies for building the documentation |
| `notebooks` | Dependencies only used in notebooks |
| `server` | Dependencies to run the TTS server |
| `bn` | Bangla G2P |
| `ja` | Japanese G2P |
| `ko` | Korean G2P |
| `zh` | Chinese G2P |
| `languages` | All language-specific dependencies |

You can install extras with one of the following commands:

```bash
pip install coqui-tts[server,ja]
pip install -e .[server,ja]
```

### Platforms

If you are on Ubuntu (Debian), you can also run following commands for installation.

```bash
make system-deps  # intended to be used on Ubuntu (Debian). Let us know if you have a different OS.
make install
```

If you are on Windows, 👑@GuyPaddock wrote installation instructions
[here](https://stackoverflow.com/questions/66726331/how-can-i-run-mozilla-tts-coqui-tts-training-with-cuda-on-a-windows-system)
(note that these are out of date, e.g. you need to have at least Python 3.9).


## Docker Image
You can also try TTS without install with the docker image.
Simply run the following command and you will be able to run TTS without installing it.

```bash
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
python3 TTS/server/server.py --list_models #To get the list of available models
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits # To start a server
```

You can then enjoy the TTS server [here](http://[::1]:5002/)
More details about the docker images (like GPU support) can be found
[here](https://coqui-tts.readthedocs.io/en/latest/docker_images.html)


## Synthesizing speech by 🐸TTS

### 🐍 Python API

#### Running a multi-speaker and multi-lingual model

```python
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
```

#### Running a single speaker model

```python
# Init TTS with the target model name
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(device)

# Run TTS
tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path=OUTPUT_PATH)

# Example voice cloning with YourTTS in English, French and Portuguese
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")
```

#### Example voice conversion

Converting the voice in `source_wav` to the voice of `target_wav`

```python
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")
tts.voice_conversion_to_file(source_wav="my/source.wav", target_wav="my/target.wav", file_path="output.wav")
```

#### Example voice cloning together with the voice conversion model.
This way, you can clone voices by using any model in 🐸TTS.

```python

tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
tts.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)
```

#### Example text to speech using **Fairseq models in ~1100 languages** 🤯.
For Fairseq models, use the following name format: `tts_models/<lang-iso_code>/fairseq/vits`.
You can find the language ISO codes [here](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html)
and learn about the Fairseq models [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms).

```python
# TTS with fairseq models
api = TTS("tts_models/deu/fairseq/vits")
api.tts_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    file_path="output.wav"
)
```

### Command-line `tts`

<!-- begin-tts-readme -->

Synthesize speech on command line.

You can either use your trained model or choose a model from the provided list.

If you don't specify any models, then it uses LJSpeech based English model.

#### Single Speaker Models

- List provided models:

  ```
  $ tts --list_models
  ```

- Get model info (for both tts_models and vocoder_models):

  - Query by type/name:
    The model_info_by_name uses the name as it from the --list_models.
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```
    For example:
    ```
    $ tts --model_info_by_name tts_models/tr/common-voice/glow-tts
    $ tts --model_info_by_name vocoder_models/en/ljspeech/hifigan_v2
    ```
  - Query by type/idx:
    The model_query_idx uses the corresponding idx from --list_models.

    ```
    $ tts --model_info_by_idx "<model_type>/<model_query_idx>"
    ```

    For example:

    ```
    $ tts --model_info_by_idx tts_models/3
    ```

  - Query info for model info by full name:
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```

- Run TTS with default models:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav
  ```

- Run TTS and pipe out the generated TTS wav file data:

  ```
  $ tts --text "Text for TTS" --pipe_out --out_path output/path/speech.wav | aplay
  ```

- Run a TTS model with its default vocoder model:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --out_path output/path/speech.wav
  ```

- Run with specific TTS and vocoder models from the list:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --vocoder_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --vocoder_name "vocoder_models/en/ljspeech/univnet" --out_path output/path/speech.wav
  ```

- Run your own TTS model (Using Griffin-Lim Vocoder):

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
  ```

- Run your own TTS and Vocoder models:

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
      --vocoder_path path/to/vocoder.pth --vocoder_config_path path/to/vocoder_config.json
  ```

#### Multi-speaker Models

- List the available speakers and choose a <speaker_id> among them:

  ```
  $ tts --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
  ```

- Run the multi-speaker TTS model with the target speaker ID:

  ```
  $ tts --text "Text for TTS." --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
  ```

- Run your own multi-speaker TTS model:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav --model_path path/to/model.pth --config_path path/to/config.json --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
  ```

### Voice Conversion Models

```
$ tts --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>" --source_wav <path/to/speaker/wav> --target_wav <path/to/reference/wav>
```

<!-- end-tts-readme -->

## Directory Structure
```
|- notebooks/       (Jupyter Notebooks for model evaluation, parameter selection and data analysis.)
|- utils/           (common utilities.)
|- TTS
    |- bin/             (folder for all the executables.)
      |- train*.py                  (train your target model.)
      |- ...
    |- tts/             (text to speech models)
        |- layers/          (model layer definitions)
        |- models/          (model definitions)
        |- utils/           (model specific utilities.)
    |- speaker_encoder/ (Speaker Encoder models.)
        |- (same)
    |- vocoder/         (Vocoder models.)
        |- (same)
```
