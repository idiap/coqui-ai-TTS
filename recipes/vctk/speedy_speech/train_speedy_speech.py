import os

from trainer import Trainer, TrainerArgs

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.speedy_speech_config import SpeedySpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS

output_path = os.path.dirname(os.path.abspath(__file__))


def main():
    dataset_config = BaseDatasetConfig(formatter="vctk", meta_file_train="", path=os.path.join(output_path, "../VCTK/"))

    audio_config = BaseAudioConfig(
        sample_rate=22050,
        do_trim_silence=True,
        trim_db=23.0,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
    )

    config = SpeedySpeechConfig(
        run_name="fast_pitch_ljspeech",
        audio=audio_config,
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        compute_input_seq_cache=True,
        precompute_num_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=50,
        print_eval=False,
        mixed_precision=False,
        min_text_len=0,
        max_text_len=500,
        min_audio_len=0,
        max_audio_len=500000,
        output_path=output_path,
        datasets=[dataset_config],
        use_speaker_embedding=True,
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
    model = ForwardTTS(config)

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
