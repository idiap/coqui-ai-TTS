import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsArgs, VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits

output_path = os.path.dirname(os.path.abspath(__file__))


def main():
    mailabs_path = "/home/julian/workspace/mailabs/**"
    dataset_paths = glob(mailabs_path)
    dataset_config = [
        BaseDatasetConfig(formatter="mailabs", meta_file_train=None, path=path, language=path.split("/")[-1])
        for path in dataset_paths
    ]

    audio_config = BaseAudioConfig(
        sample_rate=16000,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    vitsArgs = VitsArgs(
        use_language_embedding=True,
        embedded_language_dim=4,
        use_speaker_embedding=True,
        use_sdp=False,
    )

    config = VitsConfig(
        model_args=vitsArgs,
        audio=audio_config,
        run_name="vits_vctk",
        use_speaker_embedding=True,
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=0,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        use_language_weighted_sampler=True,
        print_eval=False,
        mixed_precision=False,
        min_audio_len=32 * 256 * 4,
        max_audio_len=160000,
        output_path=output_path,
        datasets=dataset_config,
        characters=CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            characters="!¡'(),-.:;¿?abcdefghijklmnopqrstuvwxyzµßàáâäåæçèéêëìíîïñòóôöùúûüąćęłńœśşźżƒабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґӧ «°±µ»$%&‘’‚“`”„",
            punctuations="!¡'(),-.:;¿? ",
            phonemes=None,
        ),
        test_sentences=[
            [
                "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "mary_ann",
                None,
                "en_US",
            ],
            [
                "Il m'a fallu beaucoup de temps pour d\u00e9velopper une voix, et maintenant que je l'ai, je ne vais pas me taire.",
                "ezwa",
                None,
                "fr_FR",
            ],
            ["Ich finde, dieses Startup ist wirklich unglaublich.", "eva_k", None, "de_DE"],
            ["Я думаю, что этот стартап действительно удивительный.", "oblomov", None, "ru_RU"],
        ],
    )

    # force the convertion of the custom characters to a config attribute
    config.from_dict(config.to_dict())

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init model
    model = Vits(config)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()


if __name__ == "__main__":
    main()
