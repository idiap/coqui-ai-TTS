# FAQ
We tried to collect common issues and questions we receive about 🐸TTS. It is
worth checking before going deeper.

## Using Coqui

### What is the Coqui fork about and how to install it?

The original Coqui package ([`TTS`](https://pypi.org/project/TTS/) on PyPI) had
its last release in December 2023. It is strongly recommended to install this
fork ([`coqui-tts`](https://pypi.org/project/coqui-tts/) on PyPI) instead for
compatibility with recent Python and dependency versions. It also includes a
large number of new features and bug fixes with many ongoing community
contributions ([full
changelog](https://github.com/idiap/coqui-ai-TTS/releases)). We welcome any
contributions and bug reports on
[GitHub](https://github.com/idiap/coqui-ai-TTS).

For general installation instructions see [this documentation](installation.md).
If you previously tried to install the original package, e.g. with `pip install
TTS`, you have to start with a new virtual environment or at least remove any
trace of the original packages, they cannot be installed together:

```bash
pip uninstall TTS trainer coqpit
pip cache purge
pip install coqui-tts
```

### Where does Coqui store downloaded models?

The path to downloaded models is printed when running `tts --list_models`.
Default locations are:

- **Linux:** `~/.local/share/tts`
- **Mac:** `~/Library/Application Support/tts`
- **Windows:** `C:\Users\<user>\AppData\Local\tts`

You can change the prefix of this `tts/` folder by setting the `XDG_DATA_HOME`
or `TTS_HOME` environment variables.

### Errors with a pre-trained model. How can I resolve this?
- Make sure you use the latest version of 🐸TTS. Each pre-trained model is only
  supported from a certain minimum version.
- If it is still problematic, post your problem on
  [Discussions](https://github.com/idiap/coqui-ai-TTS/discussions). Please give
  as many details as possible (error message, your TTS version, your TTS model
  and config.json etc.)
- If you feel like it's a bug to be fixed, then prefer GitHub issues with the
  same level of scrutiny.

## Training Coqui models

### What are the requirements of a good 🐸TTS dataset?
- [See this page](datasets/what_makes_a_good_dataset.md)

### How should I choose the right model?

```{note} This section is out-of-date and does not contain information about
more recent models available in Coqui.
```

- First, train Tacotron. It is smaller and faster to experiment with. If it performs poorly, try Tacotron2.
- Tacotron models produce the most natural voice if your dataset is not too noisy.
- If both models do not perform well and especially the attention does not align, then try AlignTTS or GlowTTS.
- If you need faster models, consider SpeedySpeech, GlowTTS or AlignTTS. Keep in mind that SpeedySpeech requires a pre-trained Tacotron or Tacotron2 model to compute text-to-speech alignments.

### How can I train my own `tts` model?

```{note} XTTS has separate fine-tuning scripts, see [here](models/xtts.md#training).
```

0. Check your dataset with notebooks in [dataset_analysis](https://github.com/idiap/coqui-ai-TTS/tree/main/notebooks/dataset_analysis) folder. Use [this notebook](https://github.com/idiap/coqui-ai-TTS/blob/main/notebooks/dataset_analysis/CheckSpectrograms.ipynb) to find the right audio processing parameters. A better set of parameters results in a better audio synthesis.

1. Write your own dataset `formatter` in `datasets/formatters.py` or [format](datasets/formatting_your_dataset) your dataset as one of the supported datasets, like LJSpeech.
    A `formatter` parses the metadata file and converts a list of training samples.

2. If you have a dataset with a different alphabet than English, you need to set your own character list in the ```config.json```.
    - If you use phonemes for training and your language is supported by
    [Espeak](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)
    or [Gruut](https://github.com/rhasspy/gruut#supported-languages), you don't need to set your character list.
    - You can use `TTS/bin/find_unique_chars.py` to get characters used in your dataset.

3. Write your own text cleaner in ```utils.text.cleaners```. It is not always necessary, except when you have a different alphabet or language-specific requirements.
    - A `cleaner` performs number and abbreviation expansion and text normalization. Basically, it converts the written text to its spoken format.
    - If you go lazy, you can try using ```basic_cleaners```.

4. Fill in a ```config.json```. Go over each parameter one by one and consider it regarding the appended explanation.
    - Check the `Coqpit` class created for your target model. Coqpit classes for `tts` models are under `TTS/tts/configs/`.
    - You just need to define fields you need/want to change in your `config.json`. For the rest, their default values are used.
    - 'sample_rate', 'phoneme_language' (if phoneme enabled), 'output_path', 'datasets', 'text_cleaner' are the fields you need to edit in most of the cases.
    - Here is a sample `config.json` for training a `GlowTTS` network.
     ```json
    {
        "model": "glow_tts",
        "batch_size": 32,
        "eval_batch_size": 16,
        "num_loader_workers": 4,
        "num_eval_loader_workers": 4,
        "run_eval": true,
        "test_delay_epochs": -1,
        "epochs": 1000,
        "text_cleaner": "english_cleaners",
        "use_phonemes": false,
        "phoneme_language": "en-us",
        "phoneme_cache_path": "phoneme_cache",
        "print_step": 25,
        "print_eval": true,
        "mixed_precision": false,
        "output_path": "recipes/ljspeech/glow_tts/",
        "test_sentences": ["Test this sentence.", "This test sentence.", "Sentence this test."],
        "datasets":[{"formatter": "ljspeech", "meta_file_train":"metadata.csv", "path": "recipes/ljspeech/LJSpeech-1.1/"}]
    }
    ```

6. Train your model.
    - SingleGPU training: ```CUDA_VISIBLE_DEVICES="0" python train_tts.py --config_path config.json```
    - MultiGPU training: ```python3 -m trainer.distribute --gpus "0,1" --script TTS/bin/train_tts.py --config_path config.json```

**Note:** You can also train your model using pure 🐍 python. Check the
[tutorial](tutorial_for_nervous_beginners.md).

### How can I train in a different language?
Check steps 2, 3, 4, 5 above.

### How can I train multi-GPUs?
Check step 5 above.

### How can I check model performance?
You can inspect model training and performance using ```tensorboard```. It will show you loss, attention alignment, model output. Go with the order below to measure the model performance.
1. Check ground truth spectrograms. If they do not look as they are supposed to, then check audio processing parameters in ```config.json```.
2. Check train and eval losses and make sure that they all decrease smoothly in time.
3. Check model spectrograms. Especially, training outputs should look similar to ground truth spectrograms after ~10K iterations.
4. Your model would not work well at test time until the attention has a near diagonal alignment. This is the sublime art of TTS training.
    - Attention should converge diagonally after ~50K iterations.
    - If attention does not converge, the probabilities are;
        - Your dataset is too noisy or small.
        - Samples are too long.
        - Batch size is too small (batch_size < 32 would be having a hard time converging)
    - You can also try other attention algorithms like 'graves', 'bidirectional_decoder', 'forward_attn'.
        - 'bidirectional_decoder' is your ultimate savior, but it trains 2x slower and demands 1.5x more GPU memory.
    - You can also try the other models like AlignTTS or GlowTTS.

### How do I know when to stop training?
There is no single objective metric to decide the end of a training since the voice quality is a subjective matter.

In our model trainings, we follow these steps;

- Check test time audio outputs, if it does not improve more.
- Check test time attention maps, if they look clear and diagonal.
- Check validation loss, if it converged and smoothly went down or started to overfit going up.
- If the answer is YES for all of the above, then test the model with a set of complex sentences. For English, you can use the `TestAttention` notebook.

Keep in mind that the approach above only validates the model robustness. It is hard to estimate the voice quality without asking the actual people.
The best approach is to pick a set of promising models and run a Mean-Opinion-Score study asking actual people to score the models.

### My model does not learn. How can I debug?
Go over the steps under "How can I check model performance?"

### Attention does not align. How can I make it work?
Check the 4th step under "How can I check model performance?"

### How can I test a trained model?
The best way is to use `tts` or `tts-server` commands. For details check [here](inference.md).

### Difference between `--continue_path` and `--restore_path`?

These are similar arguments to resume training from a checkpoint. They should be
used as follows:

`--continue_path <CHECKPOINT_DIR>`:
- To continue a training/fine-tuning run that got interrupted, e.g. because a
  job failed or reached a time limit.
- `<CHECKPOINT_DIR>` points to a folder and Coqui will resume training from the
  latest checkpoint in there and also use it to save new ones.
- Coqui restores optimizer, LR scheduler state etc. to the previous values.

`--restore_path <CHECKPOINT>`:
- `<CHECKPOINT>` points to a specific model checkpoint file, but future checkpoints
  are saved to a new folder under `output_path`.
- To start a new training run based off the given checkpoint, e.g. to
  [fine-tune](training/finetuning.md) an already trained model on a new dataset.
- Learning rate etc. are not restored from the checkpoint, you
  can specify new values in the config to change the defaults.

### My Tacotron model does not stop - I see "Decoder stopped with 'max_decoder_steps" - Stopnet does not work.
- In general, all of the above relates to the `stopnet`. It is the part of the model telling the `decoder` when to stop.
- In general, a poor `stopnet` relates to something else that is broken in your model or dataset. Especially the attention module.
- One common reason is the silent parts in the audio clips at the beginning and the ending. Check ```trim_db``` value in the config. You can find a better value for your dataset by using ```CheckSpectrogram``` notebook. If this value is too small, too much of the audio will be trimmed. If too big, then too much silence will remain. Both will curtail the `stopnet` performance.
