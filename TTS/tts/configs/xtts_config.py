from dataclasses import dataclass, field

from coqpit import Coqpit

from TTS.config.shared_configs import ModelArgs
from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class XttsAudioConfig(Coqpit):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
        dvae_sample_rate (int): The sample rate of the DVAE
    """

    sample_rate: int = 22050
    output_sample_rate: int = 24000
    dvae_sample_rate: int = 22050


@dataclass
class XttsArgs(ModelArgs):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
        gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    """

    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # constants
    duration_const: int = 102400


@dataclass
class XttsConfig(BaseTTSConfig):
    """Defines parameters for XTTS TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (XttsArgs):
            Model architecture arguments. Defaults to `XttsArgs()`.

        audio (XttsAudioConfig):
            Audio processing configuration. Defaults to `XttsAudioConfig()`.

        temperature (float):
            Temperature for the autoregressive model inference. Larger values makes predictions more creative sacrificing stability. Defaults to `0.2`.

        length_penalty (float):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length,
            which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative),
            length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.

        repetition_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. Defaults to `2.0`.

        top_p (float):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            Defaults to `0.8`.

        num_gpt_outputs (int):
            Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
            As XTTS is a probabilistic model, more samples means a higher probability of creating something "great".
            Defaults to `16`.

        gpt_cond_len (int):
            Length of the audio used as conditioning for the autoregressive  model. If audio is shorter,
            then audio length is used else the first `gpt_cond_len` secs is used. Defaults to 12 seconds.

        gpt_cond_chunk_len (int):
            Audio chunk size in seconds. Audio is split into chunks and latents are extracted for each chunk. Then
            the latents are averaged. Chunking improves the stability. It must be <= gpt_cond_len.
            If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to `4` seconds.

        max_ref_len (int):
            Maximum number of seconds of audio to be used as conditioning for the decoder. Defaults to `10`.

        sound_norm_refs (bool):
            Whether to normalize the conditioning audio. Defaults to `False`.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> config = XttsConfig()
    """

    model: str = "xtts"
    _supports_cloning: bool = True
    # model specific params
    model_args: XttsArgs = field(default_factory=XttsArgs)
    audio: XttsAudioConfig = field(default_factory=XttsAudioConfig)
    languages: list[str] = field(
        default_factory=lambda: [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh-cn",
            "hu",
            "ko",
            "ja",
            "hi",
        ]
    )

    # inference params
    temperature: float = 0.85
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.85
    num_gpt_outputs: int = 1

    # cloning
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 10
    sound_norm_refs: bool = False
