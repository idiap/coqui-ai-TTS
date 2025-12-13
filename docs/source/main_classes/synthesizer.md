# Synthesizer API

The {py:class}`TTS.utils.synthesizer.Synthesizer` provides an inference API for
TTS and voice conversion models. End users will normally use the higher-level
[Python inference API](../inference.md) instead, which offers many convenience
functions and uses the `Synthesizer` under the hood. However, you may use the
`Synthesizer` class directly for more control or additional outputs, including
timestamps.

## Usage

Load a model by name or from a checkpoint file and run synthesis:

```python
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

model_path, config_path, _ = ModelManager().download_model("tts_models/en/ljspeech/vits")

synth = Synthesizer(tts_checkpoint=model_path, tts_config_path=config_path)

wav = synth.tts("Hello World")
synth.save_wav(wav, "test_audio.wav")
```

Get additional outputs as a Python dictionary with `return_dict=True`:

```python
>>> print(synth.tts("Hello World. This is a test.", return_dict=True))

{
  'wav': [...],
  'text': 'Hello world. This is a test.',
  'segments': [
    {'id': 0, 'start': 0.0, 'end': 0.92, 'text': 'Hello world.'},
    {'id': 1, 'start': 1.37, 'end': 2.50, 'text': 'This is a test.'}
  ]
}
```


## Synthesizer class
```{eval-rst}
.. autoclass:: TTS.utils.synthesizer.Synthesizer
    :members:
```
