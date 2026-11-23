"""Search a good noise schedule for WaveGrad for a given number of inference iterations"""

import argparse
import logging
import sys
from itertools import product as cartesian_product

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from TTS.config import load_config
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import ConsoleFormatter, select_device, setup_logger
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.datasets.wavegrad_dataset import WaveGradDataset
from TTS.vocoder.models import setup_model

if __name__ == "__main__":
    setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint.")
    parser.add_argument("--config_path", type=str, help="Path to model config file.")
    parser.add_argument("--data_path", type=str, help="Path to data directory.")
    parser.add_argument("--output_path", type=str, help="path for output file including file name and extension.")
    parser.add_argument(
        "--num_iter",
        type=int,
        help="Number of model inference iterations that you like to optimize noise schedule for.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. `cpu` or `cuda`. Defaults to the best available.",
    )
    parser.add_argument("--use_cuda", action="store_true", default=None, help="DEPRECATED, use `--device`.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of datasamples used for inference.")
    parser.add_argument(
        "--search_depth",
        type=int,
        default=3,
        help="Search granularity. Increasing this increases the run-time exponentially.",
    )

    # load config
    args = parser.parse_args()
    device = select_device(args.device, use_cuda=args.use_cuda, arg_name="--use_cuda", replacement="`--device`")
    config = load_config(args.config_path)

    # setup audio processor
    ap = AudioProcessor(config.audio)

    # load dataset
    _, train_data = load_wav_data(args.data_path, 0)
    train_data = train_data[: args.num_samples]
    dataset = WaveGradDataset(
        ap=ap,
        items=train_data,
        seq_len=-1,
        hop_len=ap.hop_length,
        pad_short=config.pad_short,
        conv_pad=config.conv_pad,
        is_training=True,
        return_segments=False,
        use_noise_augment=False,
        use_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_full_clips,
        drop_last=False,
        num_workers=config.num_loader_workers,
        pin_memory=False,
    )

    # setup the model
    model = setup_model(config)
    model.to(device)

    # setup optimization parameters
    base_values = sorted(10 * np.random.uniform(size=args.search_depth))
    print(f" > base values: {base_values}")
    exponents = 10 ** np.linspace(-6, -1, num=args.num_iter)
    best_error = float("inf")
    best_schedule = None
    total_search_iter = len(base_values) ** args.num_iter
    for base in tqdm(cartesian_product(base_values, repeat=args.num_iter), total=total_search_iter):
        beta = exponents * base
        model.compute_noise_level(beta)
        for data in loader:
            mel, audio = data
            y_hat = model.inference(mel.to(device)).cpu().numpy()

            mel_hat = []
            for i in range(y_hat.shape[0]):
                m = ap.melspectrogram(y_hat[i, 0])[:, :-1]
                mel_hat.append(torch.from_numpy(m))

            mel_hat = torch.stack(mel_hat)
            mse = torch.sum((mel - mel_hat) ** 2).mean()
            if mse.item() < best_error:
                best_error = mse.item()
                best_schedule = {"beta": beta}
                print(f" > Found a better schedule. - MSE: {mse.item()}")
                np.save(args.output_path, best_schedule)
