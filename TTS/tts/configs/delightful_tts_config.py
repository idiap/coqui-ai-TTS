from dataclasses import dataclass, field

from coqpit import Coqpit

from TTS.config.shared_configs import BaseAudioConfig, ModelArgs
from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class VocoderConfig(Coqpit):
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: list[list[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: list[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: list[int] = field(default_factory=lambda: [16, 16, 4, 4])
    use_spectral_norm_discriminator: bool = False
    upsampling_rates_discriminator: list[int] = field(default_factory=lambda: [4, 4, 4, 4])
    periods_discriminator: list[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    pretrained_model_path: str | None = None


@dataclass
class DelightfulTtsAudioConfig(BaseAudioConfig):
    mel_fmax: float = 8000
    num_mels: int = 100


@dataclass
class DelightfulTtsArgs(ModelArgs):
    num_chars: int = 100
    spec_segment_size: int = 32
    n_hidden_conformer_encoder: int = 512
    n_layers_conformer_encoder: int = 6
    n_heads_conformer_encoder: int = 8
    dropout_conformer_encoder: float = 0.1
    kernel_size_conv_mod_conformer_encoder: int = 7
    kernel_size_depthwise_conformer_encoder: int = 7
    lrelu_slope: float = 0.3
    n_hidden_conformer_decoder: int = 512
    n_layers_conformer_decoder: int = 6
    n_heads_conformer_decoder: int = 8
    dropout_conformer_decoder: float = 0.1
    kernel_size_conv_mod_conformer_decoder: int = 11
    kernel_size_depthwise_conformer_decoder: int = 11
    bottleneck_size_p_reference_encoder: int = 4
    bottleneck_size_u_reference_encoder: int = 512
    ref_enc_filters_reference_encoder = [32, 32, 64, 64, 128, 128]
    ref_enc_size_reference_encoder: int = 3
    ref_enc_strides_reference_encoder = [1, 2, 1, 2, 1]
    ref_enc_pad_reference_encoder = [1, 1]
    ref_enc_gru_size_reference_encoder: int = 32
    ref_attention_dropout_reference_encoder: float = 0.2
    token_num_reference_encoder: int = 32
    predictor_kernel_size_reference_encoder: int = 5
    n_hidden_variance_adaptor: int = 512
    kernel_size_variance_adaptor: int = 5
    dropout_variance_adaptor: float = 0.5
    n_bins_variance_adaptor: int = 256
    emb_kernel_size_variance_adaptor: int = 3
    use_speaker_embedding: bool = False
    speakers_file: str = None
    d_vector_file: str = None
    speaker_embedding_channels: int = 384
    use_d_vector_file: bool = False
    d_vector_dim: int = 0
    freeze_vocoder: bool = False
    freeze_text_encoder: bool = False
    freeze_duration_predictor: bool = False
    freeze_pitch_predictor: bool = False
    freeze_energy_predictor: bool = False
    freeze_basis_vectors_predictor: bool = False
    freeze_decoder: bool = False
    length_scale: float = 1.0


@dataclass
class DelightfulTTSConfig(BaseTTSConfig):
    """
    Configuration class for the DelightfulTTS model.

    Attributes:
        model (str): Name of the model ("delightful_tts").
        audio (DelightfulTtsAudioConfig): Configuration for audio settings.
        model_args (DelightfulTtsArgs): Configuration for model arguments.
        use_attn_priors (bool): Whether to use attention priors.
        vocoder (VocoderConfig): Configuration for the vocoder.
        init_discriminator (bool): Whether to initialize the discriminator.
        steps_to_start_discriminator (int): Number of steps to start the discriminator.
        grad_clip (List[float]): Gradient clipping values.
        lr_gen (float): Learning rate for the  gan generator.
        lr_disc (float): Learning rate for the gan discriminator.
        lr_scheduler_gen (str): Name of the learning rate scheduler for the generator.
        lr_scheduler_gen_params (dict): Parameters for the learning rate scheduler for the generator.
        lr_scheduler_disc (str): Name of the learning rate scheduler for the discriminator.
        lr_scheduler_disc_params (dict): Parameters for the learning rate scheduler for the discriminator.
        scheduler_after_epoch (bool): Whether to schedule after each epoch.
        optimizer (str): Name of the optimizer.
        optimizer_params (dict): Parameters for the optimizer.
        ssim_loss_alpha (float): Alpha value for the SSIM loss.
        mel_loss_alpha (float): Alpha value for the mel loss.
        aligner_loss_alpha (float): Alpha value for the aligner loss.
        pitch_loss_alpha (float): Alpha value for the pitch loss.
        energy_loss_alpha (float): Alpha value for the energy loss.
        u_prosody_loss_alpha (float): Alpha value for the utterance prosody loss.
        p_prosody_loss_alpha (float): Alpha value for the phoneme prosody loss.
        dur_loss_alpha (float): Alpha value for the duration loss.
        char_dur_loss_alpha (float): Alpha value for the character duration loss.
        binary_align_loss_alpha (float): Alpha value for the binary alignment loss.
        binary_loss_warmup_epochs (int): Number of warm-up epochs for the binary loss.
        disc_loss_alpha (float): Alpha value for the discriminator loss.
        gen_loss_alpha (float): Alpha value for the generator loss.
        feat_loss_alpha (float): Alpha value for the feature loss.
        vocoder_mel_loss_alpha (float): Alpha value for the vocoder mel loss.
        multi_scale_stft_loss_alpha (float): Alpha value for the multi-scale STFT loss.
        multi_scale_stft_loss_params (dict): Parameters for the multi-scale STFT loss.
        return_wav (bool): Whether to return audio waveforms.
        use_weighted_sampler (bool): Whether to use a weighted sampler.
        weighted_sampler_attrs (dict): Attributes for the weighted sampler.
        weighted_sampler_multipliers (dict): Multipliers for the weighted sampler.
        r (int): Value for the `r` override.
        compute_f0 (bool): Whether to compute F0 values.
        f0_cache_path (str): Path to the F0 cache.
        attn_prior_cache_path (str): Path to the attention prior cache.
        use_speaker_embedding (bool): Whether to use speaker embedding.
        speakers_file (str): Path to the speaker file.
        speaker_embedding_channels (int): Number of channels for the speaker embedding.
    """

    model: str = "delightful_tts"

    # model specific params
    audio: DelightfulTtsAudioConfig = field(default_factory=DelightfulTtsAudioConfig)
    model_args: DelightfulTtsArgs = field(default_factory=DelightfulTtsArgs)
    use_attn_priors: bool = True

    # vocoder
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    init_discriminator: bool = True

    # optimizer
    steps_to_start_discriminator: int = 200000
    grad_clip: list[float] = field(default_factory=lambda: [1000, 1000])
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    lr_scheduler_gen: str = "ExponentialLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    scheduler_after_epoch: bool = True
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})

    # acoustic model loss params
    ssim_loss_alpha: float = 1.0
    mel_loss_alpha: float = 1.0
    aligner_loss_alpha: float = 1.0
    pitch_loss_alpha: float = 1.0
    energy_loss_alpha: float = 1.0
    u_prosody_loss_alpha: float = 0.5
    p_prosody_loss_alpha: float = 0.5
    dur_loss_alpha: float = 1.0
    char_dur_loss_alpha: float = 0.01
    binary_align_loss_alpha: float = 0.1
    binary_loss_warmup_epochs: int = 10

    # vocoder loss params
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 1.0
    feat_loss_alpha: float = 1.0
    vocoder_mel_loss_alpha: float = 10.0
    multi_scale_stft_loss_alpha: float = 2.5
    multi_scale_stft_loss_params: dict = field(
        default_factory=lambda: {
            "n_ffts": [1024, 2048, 512],
            "hop_lengths": [120, 240, 50],
            "win_lengths": [600, 1200, 240],
        }
    )

    # data loader params
    return_wav: bool = True
    use_weighted_sampler: bool = False
    weighted_sampler_attrs: dict = field(default_factory=dict)
    weighted_sampler_multipliers: dict = field(default_factory=dict)

    # overrides
    r: int = 1

    # dataset configs
    compute_f0: bool = True
    f0_cache_path: str = None
    attn_prior_cache_path: str = None

    # multi-speaker settings
    # use speaker embedding layer
    use_speaker_embedding: bool = False
    speakers_file: str = None
    speaker_embedding_channels: int = 256

    # use d-vectors
    use_d_vector_file: bool = False
    d_vector_file: str = None
    d_vector_dim: int = None

    # testing
    test_sentences: list[str] | list[list[str]] = field(
        default_factory=lambda: [
            ["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."],
            ["Be a voice, not an echo."],
            ["I'm sorry Dave. I'm afraid I can't do that."],
            ["This cake is great. It's so delicious and moist."],
            ["Prior to November 22, 1963."],
        ]
    )

    def __post_init__(self):
        # speaker embedding settings
        if self.use_speaker_embedding:
            self.model_args.use_speaker_embedding = True
        if self.speakers_file:
            self.model_args.speakers_file = self.speakers_file

        # d-vector settings
        if self.use_d_vector_file:
            self.model_args.use_d_vector_file = True
        if self.d_vector_dim is not None and self.d_vector_dim > 0:
            self.model_args.d_vector_dim = self.d_vector_dim
        if self.d_vector_file:
            self.model_args.d_vector_file = self.d_vector_file
