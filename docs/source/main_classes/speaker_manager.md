# Speaker Manager API

The {py:class}`TTS.tts.utils.speakers.SpeakerManager` organizes speaker-related data and information for 🐸TTS models. It is
especially needed for multi-speaker models and is initialized automatically when
`use_speaker_embedding` or `use_d_vector_file` are set in the config. During
training, it reads speaker names from the `speaker_name` field in your
dataset.



```{note} See [this page](../datasets/formatting_your_dataset.md#using-your-dataset-in-tts)
 on how to format your dataset for multi-speaker training.
```


## Speaker Manager
```{eval-rst}
.. automodule:: TTS.tts.utils.speakers
    :members:
```
