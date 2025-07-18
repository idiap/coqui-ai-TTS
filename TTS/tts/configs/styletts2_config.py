"""
StyleTTS 2 configuration classes.
"""

from dataclasses import dataclass, field

from coqpit import Coqpit

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class PLBertConfig(Coqpit):
    """Config for the PL-BERT model.

    PL-BERT uses HF Albert:
    https://huggingface.co/docs/transformers/en/model_doc/albert
    """

    vocab_size: int = 178
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    num_hidden_layers: int = 12
    dropout: float = 0.1


@dataclass
class StyleTTS2DecoderConfig(Coqpit):
    """Decoder config

    Args:
        decoder_type: Type of waveform generator ("hifigan" or "istftnet").
        decoder_resblock_kernel_sizes: Kernel sizes for residual blocks.
        decoder_resblock_dilation_sizes: Dilation sizes for residual blocks.
        decoder_upsample_rates: Upsampling rates.
        decoder_upsample_initial_channel: Initial channels for upsampling.
        decoder_upsample_kernel_sizes: Kernel sizes for upsampling.
    """

    generator_type: str = "istftnet"
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates: list[int] = field(default_factory=lambda: [10, 6])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [20, 12])
    gen_istft_n_fft: int = 20
    gen_istft_hop_size: int = 5


@dataclass
class StyleTTS2Args(Coqpit):
    """StyleTTS 2 model arguments.

    Args:
        # Text encoder configurations
        num_chars: Number of characters in the vocabulary.
        hidden_channels: Number of hidden channels.
        acoustic_text_encoder_layers: Number of layers in acoustic text encoder.
        plbert_config: Config for PL-BERT model.
        decoder_config: Config for decoder.

        # Style encoder configurations
        acoustic_style_encoder_layers: Number of layers in acoustic style encoder.
        prosodic_style_encoder_layers: Number of layers in prosodic style encoder.
        style_dim: Dimension of style embeddings.

        # Diffusion model configurations
        diffusion_steps: Number of diffusion steps for training.
        diffusion_inference_steps: Number of diffusion steps for inference.
        diffusion_transformer_layers: Number of transformer layers in diffusion model.
        diffusion_transformer_head_features:
        diffusion_transformer_multiplier:
        diffusion_transformer_heads: Number of transformer heads.
        diffusion_dist_mean:
        diffusion_dist_std:
        diffusion_dist_sigma_data:
        diffusion_embedding_mask_prob:

        # Duration predictor configurations
        duration_predictor_type (str): Type of duration predictor. Defaults to "differentiable".
        duration_predictor_layers (int): Number of layers in duration predictor. Defaults to 3.
        max_duration (int): Maximum duration for any phoneme. Defaults to 50.

        # SLM discriminator configurations
        slm_model_name (str): Name of the SLM model. Defaults to "microsoft/wavlm-base-plus".
        slm_discriminator_channels (int): Number of channels in SLM discriminator head. Defaults to 256.
        slm_sample_rate (int): Sample rate for SLM model. Defaults to 16000.

        # Inference configurations
        inference_noise_scale (float): Noise scale for inference. Defaults to 0.667.
        inference_length_scale (float): Length scale for inference. Defaults to 1.0.

        # Multi-speaker configurations
        use_speaker_embedding (bool): Whether to use speaker embeddings. Defaults to False.
        num_speakers (int): Number of speakers. Defaults to 0.
        speaker_embedding_dim (int): Dimension of speaker embeddings. Defaults to 256.
        use_d_vector_file (bool): Whether to use d-vector files. Defaults to False.
        d_vector_dim (int): Dimension of d-vectors. Defaults to 0.

        # Multi-lingual configurations
        use_language_embedding (bool): Whether to use language embeddings. Defaults to False.
        num_languages (int): Number of languages. Defaults to 0.
        language_embedding_dim (int): Dimension of language embeddings. Defaults to 4.
    """

    # Text encoder configurations
    num_chars: int = 178  # Match original StyleTTS2 character set
    hidden_channels: int = 512
    acoustic_text_encoder_layers: int = 3
    plbert_config: PLBertConfig = field(default_factory=PLBertConfig)
    decoder_config: StyleTTS2DecoderConfig = field(default_factory=StyleTTS2DecoderConfig)

    # Style encoder configurations
    acoustic_style_encoder_layers: int = 4
    prosodic_style_encoder_layers: int = 4
    style_dim: int = 128

    # Diffusion model configurations
    diffusion_steps: int = 4
    diffusion_inference_steps: int = 5
    diffusion_transformer_layers: int = 3
    diffusion_transformer_head_features: int = 64
    diffusion_transformer_multiplier: int = 2
    diffusion_transformer_heads: int = 8
    diffusion_dist_mean: float = -3.0
    diffusion_dist_std: float = 1.0
    diffusion_dist_sigma_data: float = 0.45731624995853165
    diffusion_embedding_mask_prob: float = 0.1

    # Duration predictor configurations
    duration_predictor_type: str = "differentiable"
    duration_predictor_layers: int = 3
    max_duration: int = 50

    # SLM discriminator configurations
    slm_model_name: str = "microsoft/wavlm-base-plus"
    slm_discriminator_channels: int = 256
    slm_sample_rate: int = 16000

    # Inference configurations
    inference_noise_scale: float = 0.667
    inference_length_scale: float = 1.0

    # Multi-speaker configurations
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speaker_embedding_dim: int = 256
    use_d_vector_file: bool = False
    d_vector_dim: int = 0

    # Multi-lingual configurations
    use_language_embedding: bool = False
    num_languages: int = 0
    language_embedding_dim: int = 4


@dataclass
class StyleTTS2Config(BaseTTSConfig):
    """Configuration for StyleTTS 2 model.

    This configuration class extends BaseTTSConfig and adds StyleTTS 2-specific
    parameters for training and inference.
    """

    model: str = "styletts2"

    # Model arguments
    model_args: StyleTTS2Args = field(default_factory=StyleTTS2Args)

    # Training configuration
    epochs: int = 200
    batch_size: int = 16
    eval_batch_size: int = 8
    mixed_precision: bool = True

    # Optimizer configuration
    optimizer: str = "AdamW"
    optimizer_params: dict = field(
        default_factory=lambda: {
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "weight_decay": 0.01,
        }
    )

    # Learning rates
    lr_gen: float = 2e-4
    lr_disc: float = 2e-4

    # Scheduler configuration
    lr_scheduler_gen: str = "ExponentialLR"
    lr_scheduler_gen_params: dict = field(
        default_factory=lambda: {
            "gamma": 0.999875,
            "last_epoch": -1,
        }
    )

    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(
        default_factory=lambda: {
            "gamma": 0.999875,
            "last_epoch": -1,
        }
    )

    # Loss configuration
    loss_weights: dict = field(
        default_factory=lambda: {
            "mel_loss": 45.0,
            "kl_loss": 1.0,
            "duration_loss": 1.0,
            "gen_loss": 1.0,
            "disc_loss": 1.0,
            "feature_loss": 2.0,
            "diffusion_loss": 1.0,
            "slm_loss": 1.0,
        }
    )

    # Training stages
    # Stage 1: Acoustic pre-training
    acoustic_pretraining_steps: int = 50000
    # Stage 2: Joint training with style diffusion and SLM discriminator
    joint_training_steps: int = 100000

    # Data configuration
    min_audio_len: int = 1024  # Minimum audio length in samples
    max_audio_len: int = 65536  # Maximum audio length in samples
    min_text_len: int = 1
    max_text_len: int = 190

    # Validation configuration
    run_eval: bool = True
    eval_split_max_size: int | None = None
    eval_split_size: float = 0.01

    # Logging configuration
    print_step: int = 50
    plot_step: int = 1000
    model_param_stats: bool = False

    # Checkpoint configuration
    save_step: int = 5000
    save_n_checkpoints: int = 5
    save_checkpoints: bool = True

    # Test sentences for evaluation
    test_sentences: list[dict] = field(
        default_factory=lambda: [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Hello world, this is a test sentence for StyleTTS 2."},
            {"text": "StyleTTS 2 achieves human-level text-to-speech synthesis."},
        ]
    )

    # Multi-speaker configuration
    use_speaker_embedding: bool = False
    speakers_file: str | None = None
    speaker_embedding_channels: int = 256
    use_d_vector_file: bool = False
    d_vector_file: list[str] | None = None
    d_vector_dim: int = 0

    # Multi-lingual configuration
    use_language_embedding: bool = False
    language_ids_file: str | None = None

    # Data loading configuration
    num_loader_workers: int = 4
    num_eval_loader_workers: int = 4
    use_noise_augment: bool = True
    start_by_longest: bool = True
    shuffle: bool = True
    drop_last: bool = True

    # Sampling configuration
    use_weighted_sampler: bool = False
    weighted_sampler_attrs: dict = field(default_factory=dict)
    weighted_sampler_multipliers: dict = field(default_factory=dict)

    # Preprocessing
    precompute_num_workers: int = 4

    # StyleTTS 2 specific configurations

    # Pre-training configuration
    use_acoustic_pretraining: bool = True
    pretraining_mel_loss_weight: float = 45.0
    pretraining_kl_loss_weight: float = 1.0
    pretraining_duration_loss_weight: float = 1.0

    # Joint training configuration
    use_joint_training: bool = True
    joint_training_start_step: int = 50000
    slm_training_start_step: int = 60000  # Start SLM discriminator training later

    # Style diffusion configuration
    diffusion_training_prob: float = 0.5  # Probability of using diffusion during training
    diffusion_noise_scale: float = 1.0
    diffusion_inference_noise_scale: float = 0.8

    # Out-of-distribution text training
    use_ood_text_training: bool = True
    ood_text_prob: float = 0.5  # Probability of using OOD text during SLM training

    # Inference configuration
    inference_noise_scale: float = 0.667
    inference_length_scale: float = 1.0

    # Voice cloning configuration
    reference_audio_max_len: int = 22050 * 10  # 10 seconds max reference audio
