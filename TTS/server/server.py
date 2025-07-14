#!flask/bin/python

"""TTS demo server."""

import argparse
import io
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs

import torch
import torchaudio

try:
    from flask import Flask, render_template, render_template_string, request, send_file
except ImportError as e:
    msg = "Server requires requires flask, use `pip install coqui-tts[server]`"
    raise ImportError(msg) from e

from TTS.api import TTS
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger
from TTS.utils.manage import ModelManager

logger = logging.getLogger(__name__)
setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="Name of one of the released vocoder models.")
    parser.add_argument("--speaker_idx", type=str, default=None, help="Default speaker ID for multi-speaker models.")

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--device", type=str, help="Device to run model on.", default="cpu")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=False, help="true to use CUDA.")
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False, help="true to enable Flask debug mode."
    )
    parser.add_argument(
        "--show_details", action=argparse.BooleanOptionalAction, default=False, help="Generate model detail page."
    )
    parser.add_argument("--language_idx", type=str, help="Default language ID for multilingual models.", default="en")
    return parser


# parse the args
args = create_argparser().parse_args()

manager = ModelManager(models_file=TTS.get_models_file_path())

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit(0)

device = args.device
if args.use_cuda:
    warnings.warn("`--use_cuda` is deprecated, use `--device cuda` instead.", DeprecationWarning, stacklevel=2)

# CASE2: load models
model_name = args.model_name if args.model_path is None else None
api = TTS(
    model_name=model_name,
    model_path=args.model_path,
    config_path=args.config_path,
    vocoder_name=args.vocoder_name,
    vocoder_path=args.vocoder_path,
    vocoder_config_path=args.vocoder_config_path,
    speakers_file_path=args.speakers_file_path,
    # language_ids_file_path=args.language_ids_file_path,
).to(device)

# TODO: set this from SpeakerManager
use_gst = api.synthesizer.tts_config.get("use_gst", False)

app = Flask(__name__)


def style_wav_uri_to_dict(style_wav: str) -> str | dict:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        show_details=args.show_details,
        use_multi_speaker=api.is_multi_speaker,
        use_multi_language=api.is_multi_lingual,
        speaker_ids=api.speakers,
        language_ids=api.languages,
        use_gst=use_gst,
        supports_cloning=api.synthesizer.tts_config.supports_cloning,
    )


@app.route("/details")
def details():
    model_config = api.synthesizer.tts_config
    vocoder_config = api.synthesizer.vocoder_config or None

    return render_template(
        "details.html",
        show_details=args.show_details,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args.__dict__,
    )


lock = Lock()


@app.route("/api/tts", methods=["GET", "POST"])
def tts():
    with lock:
        text = request.headers.get("text") or request.values.get("text", "")
        speaker_idx = (
            request.headers.get("speaker-id") or request.values.get("speaker_id", args.speaker_idx)
            if api.is_multi_speaker
            else None
        )
        # Handle empty speaker_id for voice cloning scenarios
        if speaker_idx == "":
            speaker_idx = None
        language_idx = (
            request.headers.get("language-id") or request.values.get("language_id", args.language_idx)
            if api.is_multi_lingual
            else None
        )
        # Handle empty language_id
        if language_idx == "":
            language_idx = None
        style_wav = request.headers.get("style-wav") or request.values.get("style_wav", "")
        style_wav = style_wav_uri_to_dict(style_wav)
        speaker_wav = request.headers.get("speaker-wav") or request.values.get("speaker_wav", "")

        # Basic validation
        if not text.strip():
            return {"error": "Text parameter is required"}, 400

        logger.info("Model input: %s", text)
        logger.info("Speaker idx: %s", speaker_idx)
        logger.info("Speaker wav: %s", speaker_wav)
        logger.info("Language idx: %s", language_idx)

        try:
            wavs = api.tts(
                text, speaker=speaker_idx, language=language_idx, style_wav=style_wav, speaker_wav=speaker_wav
            )
        except Exception as e:
            logger.error("TTS synthesis failed: %s", str(e))
            return {"error": f"TTS synthesis failed: {str(e)}"}, 500

        out = io.BytesIO()
        api.synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


# Basic MaryTTS compatibility layer


@app.route("/locales", methods=["GET"])
def mary_tts_api_locales():
    """MaryTTS-compatible /locales endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string("{{ locale }}\n", locale=model_details[1])


@app.route("/voices", methods=["GET"])
def mary_tts_api_voices():
    """MaryTTS-compatible /voices endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    if api.is_multi_speaker:
        return render_template_string(
            "{% for speaker in speakers %}{{ speaker }} {{ locale }} {{ gender }}\n{% endfor %}",
            speakers=api.speakers,
            locale=model_details[1],
            gender="u",
        )
    return render_template_string(
        "{{ name }} {{ locale }} {{ gender }}\n", name=model_details[3], locale=model_details[1], gender="u"
    )


@app.route("/process", methods=["GET", "POST"])
def mary_tts_api_process():
    """MaryTTS-compatible /process endpoint"""
    with lock:
        if request.method == "POST":
            data = parse_qs(request.get_data(as_text=True))
            speaker_idx = data.get("VOICE", [args.speaker_idx])[0]
            # NOTE: we ignore parameter LOCALE for now since we have only one active model
            text = data.get("INPUT_TEXT", [""])[0]
        else:
            text = request.args.get("INPUT_TEXT", "")
            speaker_idx = request.args.get("VOICE", args.speaker_idx)

        logger.info("Model input: %s", text)
        logger.info("Speaker idx: %s", speaker_idx)
        wavs = api.tts(text, speaker=speaker_idx)
        out = io.BytesIO()
        api.synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


# OpenAI-compatible Speech API
@app.route("/v1/audio/speech", methods=["POST"])
def openai_tts():
    """
    POST /v1/audio/speech
    {
      "model": "tts-1",           # ignored, defaults to args.model_name
      "voice": "alloy",           # required: a speaker ID or a file/folder for voice cloning
      "input": "Hello world!",    # required text to speak
      "response_format": "wav"    # optional: wav, opus, aac, flac, wav, pcm (alternative to format)
    }
    """
    payload = request.get_json(force=True)
    logger.info(payload)
    text = payload.get("input") or ""
    speaker_idx = payload.get("voice", args.speaker_idx) if api.is_multi_speaker else None
    fmt = payload.get("response_format", "mp3").lower()  # OpenAI default is .mp3
    speed = payload.get("speed", 1.0)
    language_idx = args.language_idx if api.is_multi_lingual else None

    speaker_wav = None
    if speaker_idx is not None:
        voice_path = Path(speaker_idx)
        if voice_path.exists() and api.synthesizer.tts_config.supports_cloning:
            speaker_wav = str(voice_path) if voice_path.is_file() else [str(w) for w in voice_path.glob("*.wav")]
            speaker_idx = None

    # here we ignore payload["model"] since its loaded at startup

    def _save_audio(waveform, sample_rate, format_args):
        buf = io.BytesIO()
        torchaudio.save(buf, waveform, sample_rate, **format_args)
        buf.seek(0)
        return buf

    def _save_pcm(waveform):
        """Raw PCM (16-bit little-endian)."""
        waveform_int16 = (waveform * 32767).to(torch.int16)
        buf = io.BytesIO()
        buf.write(waveform_int16.numpy().tobytes())
        buf.seek(0)
        return buf

    with lock:
        logger.info("Model input: %s", text)
        logger.info("Speaker idx: %s", speaker_idx)
        logger.info("Speaker wav: %s", speaker_wav)
        logger.info("Language idx: %s", language_idx)

        wavs = api.tts(text, speaker=speaker_idx, language=language_idx, speaker_wav=speaker_wav, speed=speed)
        out = io.BytesIO()
        api.synthesizer.save_wav(wavs, out)
        out.seek(0)
        waveform, sample_rate = torchaudio.load(out)

        mimetypes = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "pcm": "audio/L16",
        }

        mimetype = mimetypes.get(fmt, "audio/mpeg")
        if fmt == "wav":
            out.seek(0)
            return send_file(out, mimetype=mimetype)

        format_dispatch = {
            "mp3": lambda: _save_audio(waveform, sample_rate, {"format": "mp3"}),
            "opus": lambda: _save_audio(waveform, sample_rate, {"format": "ogg", "encoding": "opus"}),
            "aac": lambda: _save_audio(waveform, sample_rate, {"format": "mp4", "encoding": "aac"}),  # m4a container
            "flac": lambda: _save_audio(waveform, sample_rate, {"format": "flac"}),
            "pcm": lambda: _save_pcm(waveform),
        }

        # Check if format is supported
        if fmt not in format_dispatch:
            return "Unsupported format", 400

        # Generate and send file
        audio_buffer = format_dispatch[fmt]()
        return send_file(audio_buffer, mimetype=mimetype)


def main():
    app.run(debug=args.debug, host="::", port=args.port)


if __name__ == "__main__":
    main()
