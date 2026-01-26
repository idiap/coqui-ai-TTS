import os

from trainer import Trainer, TrainerArgs

from TTS.bin.compute_attention_masks import compute_attention_masks
from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.configs.forward_tts_config import ForwardTTSArgs
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.utils.manage import ModelManager

output_path = os.path.dirname(os.path.abspath(__file__))


def main():
    use_aligner = True  # learned alignments

    # init configs
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        meta_file_attn_mask=os.path.join(output_path, "../LJSpeech-1.1/metadata_attn_mask.txt") if use_aligner else "",
        path=os.path.join(output_path, "../LJSpeech-1.1/"),
    )

    audio_config = BaseAudioConfig(
        sample_rate=22050,
        do_trim_silence=True,
        trim_db=60.0,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
    )

    config = Fastspeech2Config(
        run_name="fastspeech2_ljspeech",
        model_args=ForwardTTSArgs(use_aligner=use_aligner),
        audio=audio_config,
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        compute_input_seq_cache=True,
        compute_f0=True,
        f0_cache_path=os.path.join(output_path, "f0_cache"),
        compute_energy=True,
        energy_cache_path=os.path.join(output_path, "energy_cache"),
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        precompute_num_workers=4,
        print_step=50,
        print_eval=False,
        mixed_precision=False,
        max_seq_len=500000,
        output_path=output_path,
        datasets=[dataset_config],
    )

    # compute alignments
    if not use_aligner:
        manager = ModelManager()
        model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")
        compute_attention_masks(model_path, config_path, "recipes/ljspeech/LJSpeech-1.1")

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

    # init the model
    model = ForwardTTS(config)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()


if __name__ == "__main__":
    main()
