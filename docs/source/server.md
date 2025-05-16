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

It is also possible to set a default speaker for multi-speaker models:
```bash
tts-server --model_name tts_models/en/vctk/vits --speaker_idx p376
```

Run a TTS and a vocoder model from the released model list. Note that not every vocoder is compatible with every TTS model.

```bash
tts-server --model_name "<type>/<language>/<dataset>/<model_name>" \
           --vocoder_name "<type>/<language>/<dataset>/<model_name>"
```

## Parameters

The `/api/tts` endpoint accepts the following parameters:

- `text`: Input text (required)
- `speaker-id`: Speaker ID (for multi-speaker models)
- `language-id`: Language ID (for multilingual models)
- `speaker-wav`: Reference speaker audio file path (for models with voice cloning support)
- `style-wav`: Style audio file path (for supported models)
