# :frog: TTS Demo Server
Before you use the server, make sure you
[install](https://github.com/idiap/coqui-ai-TTS/tree/dev#install-tts) :frog: TTS
properly and install the additional dependencies with `pip install
coqui-tts[server]`. Then, you can follow the steps below.

**Note:** If you install :frog:TTS using ```pip```, you can also use the ```tts-server``` endpoint on the terminal instead of the `python TTS/server/server.py` arguments.

## Example commands

List officially released models:
```bash
python TTS/server/server.py --list_models  # or
tts-server --list_models
```

Run the server with the official models:
```bash
python TTS/server/server.py --model_name tts_models/en/ljspeech/tacotron2-DCA \
       --vocoder_name vocoder_models/en/ljspeech/multiband-melgan
```

Run the server with the official models on a GPU:
```bash
CUDA_VISIBLE_DEVICES="0" python TTS/server/server.py \
    --model_name tts_models/en/ljspeech/tacotron2-DCA
    --vocoder_name vocoder_models/en/ljspeech/multiband-melgan --use_cuda
```

Run the server with a custom models:
```bash
python TTS/server/server.py --tts_checkpoint /path/to/tts/model.pth \
       --tts_config /path/to/tts/config.json \
       --vocoder_checkpoint /path/to/vocoder/model.pth \
       --vocoder_config /path/to/vocoder/config.json
```
