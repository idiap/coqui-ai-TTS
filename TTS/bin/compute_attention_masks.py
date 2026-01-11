import argparse
import logging
import os
import sys
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger

logger = logging.getLogger(__name__)


def parse_args(arg_list: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
Extract attention masks from trained Tacotron/Tacotron2 models.

These masks can be used for different purposes including training a TTS model
with a Duration Predictor.

Each attention mask is written to the same path as the input wav file with
".npy" file extension, unless --output_path is specified.
(e.g. path/bla.wav (wav file) --> path/bla.npy (attention mask))

Example run:
    CUDA_VISIBLE_DEVICE="0" python TTS/bin/compute_attention_masks.py
        --model_path /data/rw/home/Models/ljspeech-dcattn-December-14-2020_11+10AM-9d0e8c7/checkpoint_200000.pth
        --config_path /data/rw/home/Models/ljspeech-dcattn-December-14-2020_11+10AM-9d0e8c7/config.json
        --dataset_metafile metadata.csv
        --data_path /root/LJSpeech-1.1/
        --batch_size 32
        --formatter ljspeech
        --use_cuda
""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to Tacotron/Tacotron2 model file ")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to Tacotron/Tacotron2 config file.",
    )
    parser.add_argument("--output_path", type=str, help="Path to save attention masks, optional.")
    parser.add_argument(
        "--formatter",
        type=str,
        default="ljspeech",
        help="Formatter name from TTS.tts.datasets.formatters.",
    )

    parser.add_argument(
        "--dataset_metafile",
        type=str,
        default="metadata.csv",
        help="Dataset metafile inclusing file paths with transcripts.",
    )
    parser.add_argument("--data_path", type=str, help="Defines the data path.", required=True)
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=False, help="enable/disable cuda.")

    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for the model. Use batch_size=1 if you have no CUDA."
    )
    return parser.parse_args(arg_list)


def main(arg_list: list[str] | None = None) -> None:
    setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())
    args = parse_args(arg_list)
    compute_attention_masks(
        args.model_path,
        args.config_path,
        args.data_path,
        output_path=args.output_path,
        formatter=args.formatter,
        metafile=args.dataset_metafile,
        use_cuda=args.use_cuda,
        batch_size=args.batch_size,
    )
    sys.exit(0)


def compute_attention_masks(
    model_path: str | os.PathLike[Any],
    config_path: str | os.PathLike[Any],
    data_path: str | os.PathLike[Any],
    *,
    output_path: str | os.PathLike[Any] | None = None,
    formatter: str = "ljspeech",
    metafile: str = "metadata.csv",
    use_cuda: bool = True,
    batch_size: int = 16,
) -> None:
    output_path = Path(output_path if output_path else data_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    config.eval_batch_size = batch_size
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # load the model
    model = setup_model(config)
    model.load_checkpoint(config, model_path, eval=True)
    if use_cuda:
        model.cuda()

    # create data loader
    config.datasets = [BaseDatasetConfig(formatter=formatter, meta_file_train=metafile, path=data_path)]
    samples, _ = load_tts_samples(config, eval_split=False)
    loader = model.get_data_loader(config, assets=None, is_eval=True, samples=samples, verbose=True, num_gpus=0)

    # compute attentions
    file_paths = []
    with torch.inference_mode():
        for data in tqdm(loader):
            # setup input data
            text_input = data["token_id"]
            text_lengths = data["token_id_lengths"]
            mel_input = data["mel"]
            mel_lengths = data["mel_lengths"]
            item_idxs = data["item_idxs"]

            # dispatch data to GPU
            if use_cuda:
                text_input = text_input.cuda()
                text_lengths = text_lengths.cuda()
                mel_input = mel_input.cuda()
                mel_lengths = mel_lengths.cuda()

            model_outputs = model.forward(text_input, text_lengths, mel_input)

            alignments = model_outputs["alignments"].detach()
            for idx, alignment in enumerate(alignments):
                item_idx = item_idxs[idx]
                # interpolate if r > 1
                alignment = (
                    torch.nn.functional.interpolate(
                        alignment.transpose(0, 1).unsqueeze(0),
                        size=None,
                        scale_factor=model.decoder.r,
                        mode="nearest",
                        align_corners=None,
                        recompute_scale_factor=None,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
                # remove paddings
                alignment = alignment[: mel_lengths[idx], : text_lengths[idx]].cpu().numpy()
                # save output
                wav_file_path = Path(item_idx).resolve()
                rel_path = wav_file_path.relative_to(Path(data_path).resolve())
                file_path = (output_path / rel_path).with_name(wav_file_path.stem + "_attn.npy")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_paths.append([wav_file_path, file_path])
                np.save(file_path, alignment)

        output_file = output_path / "metadata_attn_mask.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for p in file_paths:
                f.write(f"{p[0]}|{p[1]}\n")
        logger.info("Metafile created: %s", output_file)


if __name__ == "__main__":
    main()
