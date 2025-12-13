(formatting_your_dataset)=
# Formatting your dataset

For training a TTS model, you need a dataset with speech recordings and
transcriptions. The speech must be divided into audio clips and each clip needs
a transcription.
It is also important to use a lossless audio file format to prevent compression artifacts. We recommend using `wav` file format.

```{note}
If you have a single audio file and you need to split it into clips, there are
different open-source tools for you. We recommend [Audacity](https://www.audacityteam.org/). It is an open-source and free audio editing software.
```

Let's assume you created the audio clips and their transcription. You can collect all your clips in a folder. Let's call this folder `wavs`.

```
/wavs
  | - audio1.wav
  | - audio2.wav
  | - audio3.wav
  ...
```

You can either create separate transcription files for each clip or create a text file that maps each audio clip to its transcription. In this file, each column must be delimited by a special character separating the audio file name, the transcription and the normalized transcription. And make sure that the delimiter is not used in the transcription text.

We recommend the following format delimited by `|`. In the following example,
`audio1`, `audio2` refer to files `audio1.wav`, `audio2.wav` etc.

```
# metadata.txt

audio1|This is my sentence.|This is my sentence.
audio2|1469 and 1470|fourteen sixty-nine and fourteen seventy
audio3|It'll be $16 sir.|It'll be sixteen dollars sir.
...
```

```{note}
If you don't have normalized transcriptions with numbers and abbreviations
spelled out, you can use the same transcription for both columns. In this
case, we recommend to use normalization later in the pipeline, either in the
text cleaner or in the phonemizer. Just make sure your metadata file still has three
columns.
```

In the end, we have the following folder structure:
```
/MyTTSDataset
      |
      | -> metadata.txt
      | -> /wavs
              | -> audio1.wav
              | -> audio2.wav
              | ...
```

The metadata format above is taken from the widely-used
[LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset. 🐸TTS already
provides tooling for LJSpeech, so if you use the same format, you can start training your models right away.

```{note}
Your dataset should have good coverage of the target language. It should cover the phonemic variety, exceptional sounds and syllables. This is extremely important for especially non-phonemic languages like English.

For more info about dataset qualities and properties check [this page](what_makes_a_good_dataset.md).
```

## Using Your Dataset in 🐸TTS

After you collect and format your dataset, you need to check two things. Whether you need a `formatter` and a `text_cleaner`. The `formatter` loads the metadata file (created above) as a list and the `text_cleaner` performs a sequence of text normalization operations that converts the raw text into the spoken representation (e.g. converting numbers to text, acronyms, and symbols to the spoken format).

If you use a different dataset format than LJSpeech or the other public datasets
that 🐸TTS supports, then you need to write your own `formatter`. See the
[list of already available formatters](#available-formatters) below.

If your dataset is in a new language or it needs special normalization steps, then you need a new `text_cleaner`.

A `formatter` returns a list of dictionaries:

```
>>> formatter(metafile_path)
[
    {"audio_file":"audio1.wav", "text":"This is my sentence.", "speaker_name":"MyDataset", "language": "lang_code"},
    {"audio_file":"audio1.wav", "text":"This is maybe a sentence.", "speaker_name":"MyDataset", "language": "lang_code"},
    ...
]
```

`audio_file` is the path to the audio file, `text` contains its transcription
and `speaker_name` is used in multi-speaker models to identify the speaker of
each sample. For single-speaker datasets `speaker_name` can simply store the
dataset name.

The purpose of a `formatter` is to parse your manifest file and load the audio file paths and transcriptions.
Then, the output is passed to the `Dataset`. It computes features from the audio signals, calls text normalization routines, and converts raw text to
phonemes if needed.

## Loading your dataset

Load one of the dataset supported by 🐸TTS. You can also consult the
[recipes](https://github.com/idiap/coqui-ai-TTS/tree/dev/recipes) Coqui already
provides for many datasets.

```python
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples


# dataset config for one of the pre-defined datasets
dataset_config = BaseDatasetConfig(
    formatter="vctk", meta_file_train="", language="en-us", path="dataset-path")
)

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)
```

Load a custom dataset with a custom formatter.

```python
from TTS.tts.datasets import load_tts_samples, register_formatter


# custom formatter implementation
def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[1]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items

register_formatter("custom_formatter_name", formatter) # Use the custom formatter name in the dataset config
dataset_config = BaseDatasetConfig(
    formatter="custom_formatter_name", meta_file_train="", language="en-us", path="dataset-path")
)
# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)
```

See {py:class}`~TTS.tts.datasets.TTSDataset`, a generic Pytorch `Dataset` implementation for the `tts` models.

See `TTS.vocoder.datasets.*`, for different `Dataset` implementations for the `vocoder` models.

See {py:class}`~TTS.utils.audio.AudioProcessor`, which includes all the audio
processing and feature extraction functions used in a `Dataset` implementation.
Feel free to add things as you need.

## Available Formatters

🐸TTS provides built-in formatters for many popular datasets. Each formatter
knows how to parse a specific dataset's metadata format. You can also use them
as a starting point to write a custom formatter for your own dataset.

```{eval-rst}
.. automodule:: TTS.tts.datasets.formatters
   :members:
   :exclude-members: Formatter, register_formatter, ljspeech_test
```
