# Synthesizer API

The {py:class}`TTS.utils.synthesizer.Synthesizer` provides an inference API for
TTS and voice conversion models. End users will normally use the higher-level
[Python inference API](../inference.md) instead, which offers many convenience
functions and uses the `Synthesizer` under the hood. However, you may use the
`Synthesizer` class directly for more control or additional outputs, including
timestamps.

## Usage

Load a model by name and run synthesis:

```python
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

model = ModelManager().load_model("tts_models/en/ljspeech/vits")

synth = Synthesizer(tts_model=model)

wav = synth.tts("Hello World")
synth.save_wav(wav, "test_audio.wav")
```

For voice cloning models, pass `voice_dir` to cache cloned voices for reuse
(see [Voice cloning](../cloning.md)):

```python
model = ModelManager().load_model("tts_models/multilingual/multi-dataset/xtts_v2")
synth = Synthesizer(tts_model=model, voice_dir="path/to/voices/")
```

Or load your own model in Coqui format from a checkpoint file:

```python
synth = Synthesizer(tts_checkpoint="path/to/model.pth", tts_config_path="path/to/config.json")
```

```{versionchanged} 0.29.0
Loading models in other formats (e.g. the original StyleTTS 2, fairseq or
OpenVoice checkpoints) from a path in the `Synthesizer` is deprecated. Load
them with `ModelManager().load_model()` and pass them via `tts_model` or
`vc_model` instead.
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
