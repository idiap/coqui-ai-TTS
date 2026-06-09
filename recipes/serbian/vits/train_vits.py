"""Vits single-speaker training recipe for Serbian.

Dataset: https://huggingface.co/datasets/daremc86/serbian_common_voice (CC-BY-4.0)

Author: Darko Milošević
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits

# set experiment paths
output_path = Path(__file__).resolve().parent
dataset_path = snapshot_download(repo_id="daremc86/serbian_common_voice", repo_type="dataset")


def main():
    # define dataset config
    dataset_config = BaseDatasetConfig(
        formatter="common_voice", meta_file_train="clean.tsv", path=dataset_path, language="sr"
    )

    # Define audio config
    audio_config = BaseAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    # Define characters config
    characters_config = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        characters="абвгдђежзијклљмнњопрстћуфхцчџшѕАБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШЅqwxyQWXY",
        punctuations="!+'(),-.:;_?/\" ",
        pad="<PAD>",
        bos="<BOS>",
        eos="<EOS>",
        blank="<BLNK>",
        phonemes=None,
    )

    # define model config
    config = VitsConfig(
        audio=audio_config,
        run_name="sr_vits",
        batch_size=16,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        save_all_best=False,
        save_best_after=10000,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        phoneme_language="sr",
        phoneme_cache_path=output_path / "phoneme_cache",
        compute_input_seq_cache=True,
        characters=characters_config,
        test_sentences=[
            "Дуга је честа оптичка појава у земљиној атмосфери, у облику једног или више обојених кружних лукова, која настаје једноструким или вишеструким ломом",
        ],
        print_step=25,
        print_eval=False,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config],
    )

    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init model
    model = Vits(config)

    # INITIALIZE THE TRAINER
    # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
    # distributed training, etc.
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    # AND... 3,2,1... 🚀
    trainer.fit()


if __name__ == "__main__":
    main()
