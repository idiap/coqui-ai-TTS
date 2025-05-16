import os
import shutil

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tests import get_tests_data_path
from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# pylint: disable=unused-variable

# create a dummy config for testing data loaders.
c = BaseTTSConfig(text_cleaner="english_cleaners", num_loader_workers=0, batch_size=2, use_noise_augment=False)
c.r = 5
c.data_path = os.path.join(get_tests_data_path(), "ljspeech/")

dataset_config_wav = BaseDatasetConfig(
    formatter="coqui",  # ljspeech_test to multi-speaker
    meta_file_train="metadata_wav.csv",
    meta_file_val=None,
    path=c.data_path,
    language="en",
)
dataset_config_mp3 = BaseDatasetConfig(
    formatter="coqui",  # ljspeech_test to multi-speaker
    meta_file_train="metadata_mp3.csv",
    meta_file_val=None,
    path=c.data_path,
    language="en",
)
dataset_config_flac = BaseDatasetConfig(
    formatter="coqui",  # ljspeech_test to multi-speaker
    meta_file_train="metadata_flac.csv",
    meta_file_val=None,
    path=c.data_path,
    language="en",
)

dataset_configs = [dataset_config_wav, dataset_config_mp3, dataset_config_flac]

ap = AudioProcessor(**c.audio)
max_loader_iter = 4

DATA_EXIST = True
if not os.path.exists(c.data_path):
    DATA_EXIST = False

print(f" > Dynamic data loader test: {DATA_EXIST}")


def _create_dataloader(batch_size, r, bgs, dataset_config, start_by_longest=False, preprocess_samples=False):
    # load dataset
    meta_data_train, meta_data_eval = load_tts_samples(dataset_config, eval_split=True, eval_split_size=0.2)
    items = meta_data_train + meta_data_eval
    tokenizer, _ = TTSTokenizer.init_from_config(c)
    dataset = TTSDataset(
        outputs_per_step=r,
        compute_linear_spec=True,
        return_wav=True,
        tokenizer=tokenizer,
        ap=ap,
        samples=items,
        batch_group_size=bgs,
        min_text_len=c.min_text_len,
        max_text_len=c.max_text_len,
        min_audio_len=c.min_audio_len,
        max_audio_len=c.max_audio_len,
        start_by_longest=start_by_longest,
    )

    # add preprocess to force the length computation
    if preprocess_samples:
        dataset.preprocess_samples()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=c.num_loader_workers,
    )
    return dataloader, dataset


@pytest.mark.parametrize("dataset_config", dataset_configs)
def test_loader(dataset_config: BaseDatasetConfig):
    batch_size = 1
    dataloader, _ = _create_dataloader(batch_size, 1, 0, dataset_config, preprocess_samples=True)
    for i, data in enumerate(dataloader):
        if i == max_loader_iter:
            break
        text_input = data["token_id"]
        _ = data["token_id_lengths"]
        speaker_name = data["speaker_names"]
        linear_input = data["linear"]
        mel_input = data["mel"]
        mel_lengths = data["mel_lengths"]
        _ = data["stop_targets"]
        _ = data["item_idxs"]
        wavs = data["waveform"]

        neg_values = text_input[text_input < 0]
        check_count = len(neg_values)

        # check basic conditions
        assert check_count == 0
        assert linear_input.shape[0] == mel_input.shape[0] == batch_size
        assert linear_input.shape[2] == ap.fft_size // 2 + 1
        assert mel_input.shape[2] == c.audio["num_mels"]
        assert wavs.shape[1] == mel_input.shape[1] * c.audio.hop_length
        assert isinstance(speaker_name[0], str)

        # make sure that the computed mels and the waveform match and correctly computed
        mel_new = ap.melspectrogram(wavs[0].squeeze().numpy())
        # guarantee that both mel-spectrograms have the same size and that we will remove waveform padding
        mel_new = mel_new[:, : mel_lengths[0]]
        ignore_seg = -(1 + c.audio.win_length // c.audio.hop_length)
        mel_diff = (mel_new[:, : mel_input.shape[1]] - mel_input[0].T.numpy())[:, 0:ignore_seg]
        assert abs(mel_diff.sum()) < 1e-4

        # check normalization ranges
        if ap.symmetric_norm:
            assert mel_input.max() <= ap.max_norm
            assert mel_input.min() >= -ap.max_norm
            assert mel_input.min() < 0
        else:
            assert mel_input.max() <= ap.max_norm
            assert mel_input.min() >= 0


def test_batch_group_shuffle():
    dataloader, dataset = _create_dataloader(2, c.r, 16, dataset_config_wav)
    last_length = 0
    frames = dataset.samples
    for i, data in enumerate(dataloader):
        if i == max_loader_iter:
            break
        mel_lengths = data["mel_lengths"]
        avg_length = mel_lengths.numpy().mean()
    dataloader.dataset.preprocess_samples()
    is_items_reordered = False
    for idx, item in enumerate(dataloader.dataset.samples):
        if item != frames[idx]:
            is_items_reordered = True
            break
    assert avg_length >= last_length
    assert is_items_reordered


def test_start_by_longest():
    """Test start_by_longest option.

    The first item of the fist batch must be longer than all the other items.
    """
    dataloader, _ = _create_dataloader(2, c.r, 0, dataset_config_wav, start_by_longest=True)
    dataloader.dataset.preprocess_samples()
    for i, data in enumerate(dataloader):
        if i == max_loader_iter:
            break
        mel_lengths = data["mel_lengths"]
        if i == 0:
            max_len = mel_lengths[0]
        print(mel_lengths)
        assert all(max_len >= mel_lengths)


def test_padding_and_spectrograms(tmp_path):
    def check_conditions(idx, linear_input, mel_input, stop_target, mel_lengths):
        assert linear_input[idx, -1].sum() != 0  # check padding
        assert linear_input[idx, -2].sum() != 0
        assert mel_input[idx, -1].sum() != 0
        assert mel_input[idx, -2].sum() != 0
        assert stop_target[idx, -1] == 1
        assert stop_target[idx, -2] == 0
        assert stop_target[idx].sum() == 1
        assert len(mel_lengths.shape) == 1
        assert mel_lengths[idx] == linear_input[idx].shape[0]
        assert mel_lengths[idx] == mel_input[idx].shape[0]

    dataloader, _ = _create_dataloader(1, 1, 0, dataset_config_wav)

    for i, data in enumerate(dataloader):
        if i == max_loader_iter:
            break
        linear_input = data["linear"]
        mel_input = data["mel"]
        mel_lengths = data["mel_lengths"]
        stop_target = data["stop_targets"]
        item_idx = data["item_idxs"]

        # check mel_spec consistency
        wav = np.asarray(ap.load_wav(item_idx[0]), dtype=np.float32)
        mel = ap.melspectrogram(wav).astype("float32")
        mel = torch.FloatTensor(mel).contiguous()
        mel_dl = mel_input[0]
        # NOTE: Below needs to check == 0 but due to an unknown reason
        # there is a slight difference between two matrices.
        # TODO: Check this assert cond more in detail.
        assert abs(mel.T - mel_dl).max() < 1e-5

        # check mel-spec correctness
        mel_spec = mel_input[0].cpu().numpy()
        wav = ap.inv_melspectrogram(mel_spec.T)
        ap.save_wav(wav, tmp_path / "mel_inv_dataloader.wav")
        shutil.copy(item_idx[0], tmp_path / "mel_target_dataloader.wav")

        # check linear-spec
        linear_spec = linear_input[0].cpu().numpy()
        wav = ap.inv_spectrogram(linear_spec.T)
        ap.save_wav(wav, tmp_path / "linear_inv_dataloader.wav")
        shutil.copy(item_idx[0], tmp_path / "linear_target_dataloader.wav")

        # check the outputs
        check_conditions(0, linear_input, mel_input, stop_target, mel_lengths)

    # Test for batch size 2
    dataloader, _ = _create_dataloader(2, 1, 0, dataset_config_wav)

    for i, data in enumerate(dataloader):
        if i == max_loader_iter:
            break
        linear_input = data["linear"]
        mel_input = data["mel"]
        mel_lengths = data["mel_lengths"]
        stop_target = data["stop_targets"]
        item_idx = data["item_idxs"]

        # set id to the longest sequence in the batch
        if mel_lengths[0] > mel_lengths[1]:
            idx = 0
        else:
            idx = 1

        # check the longer item in the batch
        check_conditions(idx, linear_input, mel_input, stop_target, mel_lengths)

        # check the other item in the batch
        assert linear_input[1 - idx, -1].sum() == 0
        assert mel_input[1 - idx, -1].sum() == 0
        assert stop_target[1, mel_lengths[1] - 1] == 1
        assert stop_target[1, mel_lengths[1] :].sum() == stop_target.shape[1] - mel_lengths[1]
        assert len(mel_lengths.shape) == 1

        # check batch zero-frame conditions (zero-frame disabled)
        # assert (linear_input * stop_target.unsqueeze(2)).sum() == 0
        # assert (mel_input * stop_target.unsqueeze(2)).sum() == 0
