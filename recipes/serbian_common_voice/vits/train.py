import os

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsArgs, VitsAudioConfig, VitsConfig

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager

# You can download serbian_common_voice dataset from https://huggingface.co/datasets/daremc86/serbian_common_voice

# set experiment paths
output_path = "/darem/serbian_cv_vits"
dataset_path = os.path.join(output_path, "serbian_common_voice")

# define dataset config
dataset_config = BaseDatasetConfig(formatter="common_voice", meta_file_train="clean.tsv", path=dataset_path, language="sr")

# Define audio config

audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

# Define VitsArgs
vits_args=VitsArgs(use_language_embedding=True, embedded_language_dim=4, use_speaker_embedding=True)

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
    model_args=vits_args,
    audio=audio_config,
    run_name="cv_vits",
    use_speaker_embedding=True,
    batch_size=32,
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
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    characters=characters_config,
    test_sentences=[
        [
            "Дуга је честа оптичка појава у земљиној атмосфери, у облику једног или више обојених кружних лукова, која настаје једноструким или вишеструким ломом",
            "MCV_Dragana",
            None,
            "sr",
    ],
    ],
    print_step=25,
    use_language_weighted_sampler=True,
    print_eval=False,
    mixed_precision=False,
    output_path=os.path.join(output_path, "output"),
    datasets=[dataset_config],
)
config.from_dict(config.to_dict())

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Initialize speker and language manager
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers
language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

# init model
model = Vits(config, ap, tokenizer, speaker_manager, language_manager)


# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
# AND... 3,2,1... 🚀
trainer.fit()