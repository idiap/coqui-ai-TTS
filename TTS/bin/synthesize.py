#!/usr/bin/env python3

"""Command line interface."""

import argparse
import contextlib
import logging
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger

logger = logging.getLogger(__name__)

description = """
Synthesize speech on the command line.

You can either use your trained model or choose a model from the provided list.

- List provided models:

  ```sh
  tts --list_models
  ```

- Get model information. Use the names obtained from `--list_models`.
  ```sh
  tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
  ```
  For example:
  ```sh
  tts --model_info_by_name tts_models/tr/common-voice/glow-tts
  tts --model_info_by_name vocoder_models/en/ljspeech/hifigan_v2
  ```

#### Single speaker models

- Run TTS with the default model (`tts_models/en/ljspeech/tacotron2-DDC`):

  ```sh
  tts --text "Text for TTS" --out_path output/path/speech.wav
  ```

- Run TTS and pipe out the generated TTS wav file data:

  ```sh
  tts --text "Text for TTS" --pipe_out --out_path output/path/speech.wav | aplay
  ```

- Run a TTS model with its default vocoder model:

  ```sh
  tts --text "Text for TTS" \\
      --model_name "<model_type>/<language>/<dataset>/<model_name>" \\
      --out_path output/path/speech.wav
  ```

  For example:

  ```sh
  tts --text "Text for TTS" \\
      --model_name "tts_models/en/ljspeech/glow-tts" \\
      --out_path output/path/speech.wav
  ```

- Run with specific TTS and vocoder models from the list. Note that not every vocoder is compatible with every TTS model.

  ```sh
  tts --text "Text for TTS" \\
      --model_name "<model_type>/<language>/<dataset>/<model_name>" \\
      --vocoder_name "<model_type>/<language>/<dataset>/<model_name>" \\
      --out_path output/path/speech.wav
  ```

  For example:

  ```sh
  tts --text "Text for TTS" \\
      --model_name "tts_models/en/ljspeech/glow-tts" \\
      --vocoder_name "vocoder_models/en/ljspeech/univnet" \\
      --out_path output/path/speech.wav
  ```

- Run your own TTS model (using Griffin-Lim Vocoder):

  ```sh
  tts --text "Text for TTS" \\
      --model_path path/to/model.pth \\
      --config_path path/to/config.json \\
      --out_path output/path/speech.wav
  ```

- Run your own TTS and Vocoder models:

  ```sh
  tts --text "Text for TTS" \\
      --model_path path/to/model.pth \\
      --config_path path/to/config.json \\
      --out_path output/path/speech.wav \\
      --vocoder_path path/to/vocoder.pth \\
      --vocoder_config_path path/to/vocoder_config.json
  ```

#### Multi-speaker models

- List the available speakers and choose a `<speaker_id>` among them:

  ```sh
  tts --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
  ```

- Run the multi-speaker TTS model with the target speaker ID:

  ```sh
  tts --text "Text for TTS." --out_path output/path/speech.wav \\
      --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
  ```

- Run your own multi-speaker TTS model:

  ```sh
  tts --text "Text for TTS" --out_path output/path/speech.wav \\
      --model_path path/to/model.pth --config_path path/to/config.json \\
      --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
  ```

#### Voice conversion models

```sh
tts --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>" \\
    --source_wav <path/to/speaker/wav> --target_wav <path/to/reference/wav>
```
"""


def parse_args(arg_list: list[str] | None) -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description=description.replace("    ```\n", ""),
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--list_models",
        action="store_true",
        help="list available pre-trained TTS and vocoder models.",
    )

    parser.add_argument(
        "--model_info_by_idx",
        type=str,
        default=None,
        help="model info using query format: <model_type>/<model_query_idx>",
    )

    parser.add_argument(
        "--model_info_by_name",
        type=str,
        default=None,
        help="model info using query format: <model_type>/<language>/<dataset>/<model_name>",
    )

    parser.add_argument("--text", type=str, default=None, help="Text to generate speech.")

    # Args for running pre-trained TTS models.
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained TTS models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        default=None,
        help="Name of one of the pre-trained  vocoder models in format <language>/<dataset>/<model_name>",
    )

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="tts_output.wav",
        help="Output wav file path.",
    )
    parser.add_argument("--use_cuda", action="store_true", help="Run model on CUDA.")
    parser.add_argument("--device", type=str, help="Device to run model on.", default="cpu")
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument(
        "--encoder_path",
        type=str,
        help="Path to speaker encoder model file.",
        default=None,
    )
    parser.add_argument("--encoder_config_path", type=str, help="Path to speaker encoder config file.", default=None)
    parser.add_argument(
        "--pipe_out",
        help="stdout the generated TTS wav file for shell pipe.",
        action="store_true",
    )

    # args for multi-speaker synthesis
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--language_ids_file_path", type=str, help="JSON file for multi-lingual model.", default=None)
    parser.add_argument(
        "--speaker_idx",
        type=str,
        help="Target speaker ID for a multi-speaker TTS model.",
        default=None,
    )
    parser.add_argument(
        "--language_idx",
        type=str,
        help="Target language ID for a multi-lingual TTS model.",
        default=None,
    )
    parser.add_argument(
        "--speaker_wav",
        nargs="+",
        help="wav file(s) to condition a multi-speaker TTS model with a Speaker Encoder. You can give multiple file paths. The d_vectors is computed as their average.",
        default=None,
    )
    parser.add_argument("--gst_style", help="Wav path file for GST style reference.", default=None)
    parser.add_argument(
        "--capacitron_style_wav", type=str, help="Wav path file for Capacitron prosody reference.", default=None
    )
    parser.add_argument("--capacitron_style_text", type=str, help="Transcription of the reference.", default=None)
    parser.add_argument(
        "--list_speaker_idxs",
        help="List available speaker ids for the defined multi-speaker model.",
        action="store_true",
    )
    parser.add_argument(
        "--list_language_idxs",
        help="List available language ids for the defined multi-lingual model.",
        action="store_true",
    )
    # aux args
    parser.add_argument(
        "--reference_wav",
        type=str,
        help="Reference wav file to convert in the voice of the speaker_idx or speaker_wav",
        default=None,
    )
    parser.add_argument(
        "--reference_speaker_idx",
        type=str,
        help="speaker ID of the reference_wav speaker (If not provided the embedding will be computed using the Speaker Encoder).",
        default=None,
    )
    parser.add_argument(
        "--progress_bar",
        action=argparse.BooleanOptionalAction,
        help="Show a progress bar for the model download.",
        default=True,
    )

    # voice conversion args
    parser.add_argument(
        "--source_wav",
        type=str,
        default=None,
        help="Original audio file to convert into the voice of the target_wav",
    )
    parser.add_argument(
        "--target_wav",
        type=str,
        nargs="*",
        default=None,
        help="Audio file(s) of the target voice into which to convert the source_wav",
    )

    parser.add_argument(
        "--voice_dir",
        type=str,
        default=None,
        help="Voice dir for tortoise model",
    )

    args = parser.parse_args(arg_list)

    # print the description if either text or list_models is not set
    check_args = [
        args.text,
        args.list_models,
        args.list_speaker_idxs,
        args.list_language_idxs,
        args.reference_wav,
        args.model_info_by_idx,
        args.model_info_by_name,
        args.source_wav,
        args.target_wav,
    ]
    if not any(check_args):
        parser.parse_args(["-h"])
    return args


def main(arg_list: list[str] | None = None) -> None:
    """Entry point for `tts` command line interface."""
    args = parse_args(arg_list)
    stream = sys.stderr if args.pipe_out else sys.stdout
    setup_logger("TTS", level=logging.INFO, stream=stream, formatter=ConsoleFormatter())

    pipe_out = sys.stdout if args.pipe_out else None

    with contextlib.redirect_stdout(None if args.pipe_out else sys.stdout):
        # Late-import to make things load faster
        from TTS.api import TTS
        from TTS.utils.manage import ModelManager

        # load model manager
        manager = ModelManager(models_file=TTS.get_models_file_path(), progress_bar=args.progress_bar)

        tts_path = None
        tts_config_path = None
        speakers_file_path = None
        language_ids_file_path = None
        vocoder_path = None
        vocoder_config_path = None
        encoder_path = None
        encoder_config_path = None
        vc_path = None
        vc_config_path = None
        model_dir = None

        # 1) List pre-trained TTS models
        if args.list_models:
            manager.list_models()
            sys.exit(0)

        # 2) Info about pre-trained TTS models (without loading a model)
        if args.model_info_by_idx:
            model_query = args.model_info_by_idx
            manager.model_info_by_idx(model_query)
            sys.exit(0)

        if args.model_info_by_name:
            model_query_full_name = args.model_info_by_name
            manager.model_info_by_full_name(model_query_full_name)
            sys.exit(0)

        # 3) Load a model for further info or TTS/VC
        device = args.device
        if args.use_cuda:
            device = "cuda"
        # A local model will take precedence if specified via modeL_path
        model_name = args.model_name if args.model_path is None else None
        api = TTS(
            model_name=model_name,
            model_path=args.model_path,
            config_path=args.config_path,
            vocoder_name=args.vocoder_name,
            vocoder_path=args.vocoder_path,
            vocoder_config_path=args.vocoder_config_path,
            encoder_path=args.encoder_path,
            encoder_config_path=args.encoder_config_path,
            speakers_file_path=args.speakers_file_path,
            language_ids_file_path=args.language_ids_file_path,
            progress_bar=args.progress_bar,
        ).to(device)

        # query speaker ids of a multi-speaker model.
        if args.list_speaker_idxs:
            if not api.is_multi_speaker:
                logger.info("Model only has a single speaker.")
                sys.exit(0)
            logger.info(
                "Available speaker ids: (Set --speaker_idx flag to one of these values to use the multi-speaker model."
            )
            logger.info(api.speakers)
            sys.exit(0)

        # query language ids of a multi-lingual model.
        if args.list_language_idxs:
            if not api.is_multi_lingual:
                logger.info("Monolingual model.")
                sys.exit(0)
            logger.info(
                "Available language ids: (Set --language_idx flag to one of these values to use the multi-lingual model."
            )
            logger.info(api.languages)
            sys.exit(0)

        # RUN THE SYNTHESIS
        if args.text:
            logger.info("Text: %s", args.text)

        if args.text is not None:
            api.tts_to_file(
                text=args.text,
                speaker=args.speaker_idx,
                language=args.language_idx,
                speaker_wav=args.speaker_wav,
                pipe_out=pipe_out,
                file_path=args.out_path,
                reference_wav=args.reference_wav,
                style_wav=args.capacitron_style_wav,
                style_text=args.capacitron_style_text,
                reference_speaker_name=args.reference_speaker_idx,
                voice_dir=args.voice_dir,
            )
            logger.info("Saved TTS output to %s", args.out_path)
        elif args.source_wav is not None and args.target_wav is not None:
            api.voice_conversion_to_file(
                source_wav=args.source_wav,
                target_wav=args.target_wav,
                file_path=args.out_path,
                speaker=args.speaker_idx,
                voice_dir=args.voice_dir,
                pipe_out=pipe_out,
            )
            logger.info("Saved VC output to %s", args.out_path)
        sys.exit(0)


if __name__ == "__main__":
    main()
