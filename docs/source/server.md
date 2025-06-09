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

Run a TTS model saved locally using GPU (cuda), supplying a language code for the model (default "en" for English):
```bash
tts-server --use_cuda --model_path C:\directory_path\to\model\directory --config_path C:\file_path\to\model\config.json --speakers_file_path C:\file_path\to\model\speakers.json --language_id en 
```

Or using server.py directly, with language set to spanish and optional --device argument to set specific cuda device:
```
python C:\absolute\path\to\server.py --device cuda:0 --model_path C:\directory_path\to\model\directory --config_path C:\file_path\to\model\config.json --speakers_file_path C:\file_path\to\model\speakers.json --language_id es
```

## Parameters

The `/api/tts` endpoint accepts the following parameters:

- `text`: Input text (required)
- `speaker-id`: Speaker ID (for multi-speaker models)
- `language-id`: Language ID (for multilingual models)
- `speaker-wav`: Reference speaker audio file path (for models with voice cloning support)
- `style-wav`: Style audio file path (for supported models)

There is also a basic openai-compatible server endpoint at `/v1/audio/speech` which accepts the parameters in the basic OpenAI speech spec ( https://platform.openai.com/docs/api-reference/audio )

- `model`: A string representing the model name (this is optional and ignored because the model is loaded with the server startup)
- `input`: Input text (required)
- `voice`: Can be one of the following: a string representing a Speaker ID in a multi-speaker TTS model, e.g., "Craig Gutsy" for XTTS2, a reference speaker audio file path (for models with voice cloning support) or a reference speaker directory path to a directory containing multiple audio files for that speaker (for models with voice cloning support)
- `speed`: Optional, float value between 0.25-4.0 (default 1.0)
- `response_format`: Optional, expected format of audio for response (defaults to mp3) (Options: "wav", "mp3", "opus", "aac", "flac", "pcm")

When using the openai-compatible server endpoint, you should specify the language (if other than English) when running the server with the command line argument --language_id `langauge_code` for multilingual models
