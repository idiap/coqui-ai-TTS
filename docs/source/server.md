# Demo server

![server.gif](https://github.com/idiap/coqui-ai-TTS/raw/main/images/demo_server.gif)

You can boot up a demo 🐸TTS server to run an inference with your models (make
sure to install the additional dependencies with `pip install coqui-tts[server]`).
Note that the server is not optimized for performance.

The demo server provides pretty much the same interface as the CLI command.

```bash
tts-server -h # see the help
tts-server --list_models  # list the available models.
```

Run a TTS model, from the release models list, with its default vocoder.
If the model you choose is a multi-speaker or multilingual TTS model, you can
select different speakers and languages on the Web interface (default URL:
http://localhost:5002) and synthesize speech.

```bash
tts-server --model_name "<type>/<language>/<dataset>/<model_name>"
```

It is also possible to set a default speaker for multi-speaker models, so you
don't have to add it in every request (although you can overwrite it):
```bash
tts-server --model_name tts_models/en/vctk/vits --speaker_idx p376
```

And a default language ID for multilingual models (defaults to `en` for English):
```bash
tts-server --model_name tts_models/multilingual/multi-dataset/xtts_v2 --language_idx es
```

Run a TTS and a vocoder model from the released model list. Note that not every vocoder is compatible with every TTS model.

```bash
tts-server --model_name "<type>/<language>/<dataset>/<model_name>" \
           --vocoder_name "<type>/<language>/<dataset>/<model_name>"
```

## Parameters

### Default endpoint

The `/api/tts` endpoint accepts the following parameters:

- `text`: Input text (required).
- `speaker-id`: Speaker ID (for multi-speaker models).
- `language-id`: Language ID (for multilingual models).
- `speaker-wav`: Reference speaker audio file path (for models with voice cloning support).
- `style-wav`: Style audio file path (for supported models).

### OpenAI-compatible endpoint

There is also a basic
[OpenAI-compatible](https://platform.openai.com/docs/api-reference/audio) server
endpoint at `/v1/audio/speech`, which accepts these parameters:

- `model`: A string representing the model name (this is optional and ignored
  because the model is loaded with the server startup).
- `input`: Input text (required).
- `voice`: Can be one of the following: a string representing a Speaker ID in a
  multi-speaker TTS model, e.g., "Craig Gutsy" for XTTS2, a reference speaker
  audio file path (for models with voice cloning support) or a reference
  speaker directory path to a directory containing multiple audio files for
  that speaker (for models with voice cloning support).
- `speed`: Optional, float (defaults to `1.0`).
- `response_format`: Optional, expected format of audio for response (defaults
  to `mp3`). Options: `wav`, `mp3`, `opus`, `aac`, `flac`, `pcm`.

When using the OpenAI-compatible endpoint, you should specify the language (if
other than English) when running the server with the command line argument
`--language_idx <language_code>` for multilingual models.
