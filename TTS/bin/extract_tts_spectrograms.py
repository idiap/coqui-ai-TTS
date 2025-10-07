#!/usr/bin/env python3
"""Extract Mel spectrograms with teacher forcing."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer.generic_utils import count_parameters

from TTS.config import load_config
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.datasets import TTSDataset, load_tts_samples
from TTS.tts.models import setup_model
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import quantize
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger

use_cuda = torch.cuda.is_available()


def parse_args(arg_list: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Extract mel spectrograms from audio using teacher forcing with a trained TTS model.

This script loads a trained TTS model and extracts mel spectrograms by running the model with teacher forcing.
This is useful for analyzing model predictions, creating training data for downstream models, or debugging
model behavior. Supports Tacotron, Tacotron2, and Glow-TTS models.

The script will create subdirectories in the output path:
  - mel/: Extracted mel spectrograms (.npy files)
  - wav/: Original audio files (if --save_audio is enabled)
  - wav_gl/: Griffin-Lim reconstructed audio from mels (if --debug is enabled)
  - quant/: Quantized audio files (if --quantize_bits > 0)""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example usage:
  python extract_tts_spectrograms.py \\
    --config_path /path/to/config.json \\
    --checkpoint_path /path/to/checkpoint.pth \\
    --output_path /path/to/output""",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the model configuration file (JSON) used during training. "
        "This config defines the model architecture, audio parameters, and dataset settings.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the trained model checkpoint file (.pth) to be loaded for inference.",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Directory path where extracted mel spectrograms and optional audio files will be saved. "
        "Subdirectories will be created automatically.",
        default="output_extract_tts_spectrograms",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode: saves Griffin-Lim reconstructed audio files from the extracted mel spectrograms "
        "to wav_gl/ subdirectory for quality inspection.",
    )
    parser.add_argument(
        "--save_audio",
        default=False,
        action="store_true",
        help="Save the original audio files to the wav/ subdirectory alongside the extracted mel spectrograms.",
    )
    parser.add_argument(
        "--quantize_bits",
        type=int,
        default=0,
        help="Bit depth for audio quantization (e.g., 8, 16). If set to a non-zero value, saves quantized versions "
        "of audio files to the quant/ subdirectory. Set to 0 (default) to disable quantization.",
    )
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        help="Include evaluation split in processing. When enabled (default), processes both training and evaluation "
        "samples. Use --no-eval to process only training samples.",
        default=True,
    )
    return parser.parse_args(arg_list)


def setup_loader(config: BaseTTSConfig, ap: AudioProcessor, r, speaker_manager: SpeakerManager, samples) -> DataLoader:
    tokenizer, _ = TTSTokenizer.init_from_config(config)
    dataset = TTSDataset(
        outputs_per_step=r,
        compute_linear_spec=False,
        samples=samples,
        tokenizer=tokenizer,
        ap=ap,
        batch_group_size=0,
        min_text_len=config.min_text_len,
        max_text_len=config.max_text_len,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        phoneme_cache_path=config.phoneme_cache_path,
        precompute_num_workers=0,
        use_noise_augment=False,
        speaker_id_mapping=speaker_manager.name_to_id if config.use_speaker_embedding else None,
        d_vector_mapping=speaker_manager.embeddings if config.use_d_vector_file else None,
    )

    if config.use_phonemes and config.compute_input_seq_cache:
        # precompute phonemes to have a better estimate of sequence lengths.
        dataset.compute_input_seq(config.num_loader_workers)
    dataset.preprocess_samples()

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
        sampler=None,
        num_workers=config.num_loader_workers,
        pin_memory=False,
    )


def format_data(data):
    # setup input data
    text_input = data["token_id"]
    text_lengths = data["token_id_lengths"]
    mel_input = data["mel"]
    mel_lengths = data["mel_lengths"]
    item_idx = data["item_idxs"]
    d_vectors = data["d_vectors"]
    speaker_ids = data["speaker_ids"]
    attn_mask = data["attns"]
    avg_text_length = torch.mean(text_lengths.float())
    avg_spec_length = torch.mean(mel_lengths.float())

    # dispatch data to GPU
    if use_cuda:
        text_input = text_input.cuda(non_blocking=True)
        text_lengths = text_lengths.cuda(non_blocking=True)
        mel_input = mel_input.cuda(non_blocking=True)
        mel_lengths = mel_lengths.cuda(non_blocking=True)
        if speaker_ids is not None:
            speaker_ids = speaker_ids.cuda(non_blocking=True)
        if d_vectors is not None:
            d_vectors = d_vectors.cuda(non_blocking=True)
        if attn_mask is not None:
            attn_mask = attn_mask.cuda(non_blocking=True)
    return (
        text_input,
        text_lengths,
        mel_input,
        mel_lengths,
        speaker_ids,
        d_vectors,
        avg_text_length,
        avg_spec_length,
        attn_mask,
        item_idx,
    )


@torch.inference_mode()
def inference(
    model_name: str,
    model: BaseTTS,
    ap: AudioProcessor,
    text_input,
    text_lengths,
    mel_input,
    mel_lengths,
    speaker_ids=None,
    d_vectors=None,
) -> np.ndarray:
    if model_name == "glow_tts":
        speaker_c = None
        if speaker_ids is not None:
            speaker_c = speaker_ids
        elif d_vectors is not None:
            speaker_c = d_vectors
        outputs = model.inference_with_MAS(
            text_input,
            text_lengths,
            mel_input,
            mel_lengths,
            aux_input={"d_vectors": speaker_c, "speaker_ids": speaker_ids},
        )
        model_output = outputs["model_outputs"]
        return model_output.detach().cpu().numpy()

    if "tacotron" in model_name:
        aux_input = {"speaker_ids": speaker_ids, "d_vectors": d_vectors}
        outputs = model(text_input, text_lengths, mel_input, mel_lengths, aux_input)
        postnet_outputs = outputs["model_outputs"]
        # normalize tacotron output
        if model_name == "tacotron":
            mel_specs = []
            postnet_outputs = postnet_outputs.data.cpu().numpy()
            for b in range(postnet_outputs.shape[0]):
                postnet_output = postnet_outputs[b]
                mel_specs.append(torch.FloatTensor(ap.out_linear_to_mel(postnet_output.T).T))
            return torch.stack(mel_specs).cpu().numpy()
        if model_name == "tacotron2":
            return postnet_outputs.detach().cpu().numpy()
    msg = f"Model not supported: {model_name}"
    raise ValueError(msg)


def extract_spectrograms(
    model_name: str,
    data_loader: DataLoader,
    model: BaseTTS,
    ap: AudioProcessor,
    output_path: Path,
    quantize_bits: int = 0,
    save_audio: bool = False,
    debug: bool = False,
    metadata_name: str = "metadata.txt",
) -> None:
    model.eval()
    export_metadata = []
    for _, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        # format data
        (
            text_input,
            text_lengths,
            mel_input,
            mel_lengths,
            speaker_ids,
            d_vectors,
            _,
            _,
            _,
            item_idx,
        ) = format_data(data)

        model_output = inference(
            model_name,
            model,
            ap,
            text_input,
            text_lengths,
            mel_input,
            mel_lengths,
            speaker_ids,
            d_vectors,
        )

        (output_path / "mel").mkdir(exist_ok=True, parents=True)
        for idx in range(text_input.shape[0]):
            wav_file_path = Path(item_idx[idx])
            wav = ap.load_wav(wav_file_path)

            # quantize and save wav
            if quantize_bits > 0:
                wavq = quantize(x=wav, quantize_bits=quantize_bits)
                (output_path / "quant").mkdir(exist_ok=True)
                np.save(output_path / "quant" / wav_file_path.stem, wavq)

            # save TTS mel
            mel = model_output[idx]
            mel_length = mel_lengths[idx]
            mel = mel[:mel_length, :].T
            np.save(output_path / "mel" / wav_file_path.stem, mel)

            export_metadata.append(output_path / "mel" / wav_file_path.stem)
            if save_audio:
                (output_path / "wav").mkdir(exist_ok=True)
                ap.save_wav(wav, output_path / "wav" / f"{wav_file_path.stem}.wav")

            if debug:
                wav_gl = ap.inv_melspectrogram(mel)
                (output_path / "wav_gl").mkdir(exist_ok=True)
                ap.save_wav(wav_gl, output_path / "wav_gl" / f"{wav_file_path.stem}.wav")

    with (output_path / metadata_name).open("w") as f:
        for path in export_metadata:
            f.write(f"{path}.npy\n")


def main(arg_list: list[str] | None = None) -> None:
    setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())
    args = parse_args(arg_list)
    config = load_config(args.config_path)
    config.audio.trim_silence = False

    # Audio processor
    ap = AudioProcessor(**config.audio)

    # load data instances
    meta_data_train, meta_data_eval = load_tts_samples(
        config.datasets,
        eval_split=args.eval,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # use eval and training partitions
    meta_data = meta_data_train + meta_data_eval

    # init speaker manager
    speaker_manager = SpeakerManager.init_from_config(config)

    # setup model
    model = setup_model(config)

    # restore model
    model.load_checkpoint(config, args.checkpoint_path, eval=True)

    if use_cuda:
        model.cuda()

    num_params = count_parameters(model)
    print(f"\n > Model has {num_params} parameters", flush=True)
    # set r
    r = 1 if config.model.lower() == "glow_tts" else model.decoder.r
    own_loader = setup_loader(config, ap, r, speaker_manager, meta_data)

    extract_spectrograms(
        config.model.lower(),
        own_loader,
        model,
        ap,
        Path(args.output_path),
        quantize_bits=args.quantize_bits,
        save_audio=args.save_audio,
        debug=args.debug,
        metadata_name="metadata.txt",
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
