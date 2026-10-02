from dataclasses import dataclass, field

from coqpit import Coqpit

from TTS.config.shared_configs import ModelArgs
from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class TortoiseAudioConfig(Coqpit):
    sample_rate: int = 22050
    diffusion_sample_rate: int = 24000
    output_sample_rate: int = 24000


@dataclass
class TortoiseArgs(ModelArgs):
    """A dataclass to represent Tortoise model arguments that define the model structure.

    Args:
        autoregressive_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        high_vram (bool, optional): Deprecated, has no effect.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For UnifiedVoice model:
        ar_max_mel_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        ar_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        ar_max_conditioning_inputs (int, optional): The maximum conditioning inputs for the autoregressive model. Defaults to 2.
        ar_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        ar_model_dim (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        ar_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        ar_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        ar_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        ar_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        ar_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.

        For DiffTTS model:
        diff_model_channels (int, optional): The number of channels for the DiffTTS model. Defaults to 1024.
        diff_num_layers (int, optional): The number of layers for the DiffTTS model. Defaults to 10.
        diff_in_channels (int, optional): The input channels for the DiffTTS model. Defaults to 100.
        diff_out_channels (int, optional): The output channels for the DiffTTS model. Defaults to 200.
        diff_in_latent_channels (int, optional): The input latent channels for the DiffTTS model. Defaults to 1024.
        diff_in_tokens (int, optional): The input tokens for the DiffTTS model. Defaults to 8193.
        diff_dropout (int, optional): The dropout percentage for the DiffTTS model. Defaults to 0.
        diff_use_fp16 (bool, optional): Whether to use fp16 for the DiffTTS model. Defaults to False.
        diff_num_heads (int, optional): The number of heads for the DiffTTS model. Defaults to 16.
        diff_layer_drop (int, optional): The layer dropout percentage for the DiffTTS model. Defaults to 0.
        diff_unconditioned_percentage (int, optional): The percentage of unconditioned inputs for the DiffTTS model. Defaults to 0.

        For ConditionalLatentVariablePerseq model:
        clvp_dim_text (int): The dimension of the text input for the CLVP module. Defaults to 768.
        clvp_dim_speech (int): The dimension of the speech input for the CLVP module. Defaults to 768.
        clvp_dim_latent (int): The dimension of the latent representation for the CLVP module. Defaults to 768.
        clvp_num_text_tokens (int): The number of text tokens used by the CLVP module. Defaults to 256.
        clvp_text_enc_depth (int): The depth of the text encoder in the CLVP module. Defaults to 20.
        clvp_text_seq_len (int): The maximum sequence length of the text input for the CLVP module. Defaults to 350.
        clvp_text_heads (int): The number of attention heads used by the text encoder in the CLVP module. Defaults to 12.
        clvp_num_speech_tokens (int): The number of speech tokens used by the CLVP module. Defaults to 8192.
        clvp_speech_enc_depth (int): The depth of the speech encoder in the CLVP module. Defaults to 20.
        clvp_speech_heads (int): The number of attention heads used by the speech encoder in the CLVP module. Defaults to 12.
        clvp_speech_seq_len (int): The maximum sequence length of the speech input for the CLVP module. Defaults to 430.
        clvp_use_xformers (bool): A flag indicating whether the model uses transformers in the CLVP module. Defaults to True.
        duration_const (int): A constant value used in the model. Defaults to 102400.
    """

    autoregressive_batch_size: int = 1
    enable_redaction: bool = False
    high_vram: bool = False
    kv_cache: bool = True
    num_chars: int = 255

    # UnifiedVoice params
    ar_max_mel_tokens: int = 604
    ar_max_text_tokens: int = 402
    ar_max_conditioning_inputs: int = 2
    ar_layers: int = 30
    ar_model_dim: int = 1024
    ar_heads: int = 16
    ar_number_text_tokens: int = 255
    ar_start_text_token: int = 255
    ar_checkpointing: bool = False
    ar_train_solo_embeddings: bool = False

    # DiffTTS params
    diff_model_channels: int = 1024
    diff_num_layers: int = 10
    diff_in_channels: int = 100
    diff_out_channels: int = 200
    diff_in_latent_channels: int = 1024
    diff_in_tokens: int = 8193
    diff_dropout: int = 0
    diff_use_fp16: bool = False
    diff_num_heads: int = 16
    diff_layer_drop: int = 0
    diff_unconditioned_percentage: int = 0

    # clvp params
    clvp_dim_text: int = 768
    clvp_dim_speech: int = 768
    clvp_dim_latent: int = 768
    clvp_num_text_tokens: int = 256
    clvp_text_enc_depth: int = 20
    clvp_text_seq_len: int = 350
    clvp_text_heads: int = 12
    clvp_num_speech_tokens: int = 8192
    clvp_speech_enc_depth: int = 20
    clvp_speech_heads: int = 12
    clvp_speech_seq_len: int = 430
    clvp_use_xformers: bool = True
    # constants
    duration_const: int = 102400


@dataclass
class TortoiseConfig(BaseTTSConfig):
    """Defines parameters for Tortoise TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (TortoiseArgs):
            Model architecture arguments. Defaults to `TortoiseArgs()`.

        audio (TortoiseAudioConfig):
            Audio processing configuration. Defaults to `TortoiseAudioConfig()`.

        temperature (float):
            Temperature for the autoregressive model inference. Larger values makes predictions more creative sacrificing stability. Defaults to `0.2`.

        length_penalty (float):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length,
            which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative),
            length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.

        reperation_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. Defaults to `2.0`.

        top_p (float):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            Defaults to `0.8`.

        cond_free_k (float):
            Knob that determines how to balance the conditioning free signal with the conditioning-present signal. [0,inf].
            As cond_free_k increases, the output becomes dominated by the conditioning-free signal.
            Formula is: output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k. Defaults to `2.0`.

        diffusion_temperature (float):
            Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0
            are the "mean" prediction of the diffusion network and will sound bland and smeared.
            Defaults to `1.0`.

        num_autoregressive_samples (int):
            Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
            As Tortoise is a probabilistic model, more samples means a higher probability of creating something "great".
            Defaults to `16`.

        diffusion_iterations (int):
            Number of diffusion steps to perform. [0,4000]. More steps means the network has more chances to iteratively refine
            the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better,
            however. Defaults to `30`.

        sampler (str):
            Diffusion sampler to be used. `ddim` or `dpm++2m`. Defaults to `ddim`.
    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.tts.configs.tortoise_config import TortoiseConfig
        >>> config = TortoiseConfig()
    """

    model: str = "tortoise"
    _supports_cloning: bool = True
    # model specific params
    model_args: TortoiseArgs = field(default_factory=TortoiseArgs)
    audio: TortoiseAudioConfig = field(default_factory=TortoiseAudioConfig)

    # settings
    temperature: float = 0.2
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_p: float = 0.8
    cond_free_k: float = 2.0
    diffusion_temperature: float = 1.0

    # inference params
    num_autoregressive_samples: int = 16
    diffusion_iterations: int = 30
    sampler: str = "ddim"
