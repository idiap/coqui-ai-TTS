# Demo server

![server.gif](https://github.com/idiap/coqui-ai-TTS/raw/main/images/demo_server.gif)

You can boot up a demo 🐸TTS server to run an inference with your models (make
sure to install the additional dependencies with `pip install coqui-tts[server]`).
Note that the server is not optimized for performance and does not support all
Coqui models yet.

The demo server provides pretty much the same interface as the CLI command.

```bash
tts-server -h # see the help
tts-server --list_models  # list the available models.
```

Run a TTS model, from the release models list, with its default vocoder.
If the model you choose is a multi-speaker TTS model, you can select different speakers on the Web interface and synthesize
speech.

```bash
tts-server --model_name "<type>/<language>/<dataset>/<model_name>"
```

Run a TTS and a vocoder model from the released model list. Note that not every vocoder is compatible with every TTS model.

```bash
tts-server --model_name "<type>/<language>/<dataset>/<model_name>" \
           --vocoder_name "<type>/<language>/<dataset>/<model_name>"
```
