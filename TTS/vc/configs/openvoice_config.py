from dataclasses import dataclass, field

from coqpit import Coqpit

from TTS.vc.configs.shared_configs import BaseVCConfig


@dataclass
class OpenVoiceAudioConfig(Coqpit):
    """Audio configuration

    Args:
        input_sample_rate (int):
            The sampling rate of the input waveform.

        output_sample_rate (int):
            The sampling rate of the output waveform.

        fft_size (int):
            The length of the filter.

        hop_length (int):
            The hop length.

        win_length (int):
            The window length.
    """

    input_sample_rate: int = field(default=22050)
    output_sample_rate: int = field(default=22050)
    fft_size: int = field(default=1024)
    hop_length: int = field(default=256)
    win_length: int = field(default=1024)


@dataclass
class OpenVoiceArgs(Coqpit):
    """OpenVoice model arguments.

    zero_g (bool):
        Whether to zero the gradients.

    inter_channels (int):
        The number of channels in the intermediate layers.

    hidden_channels (int):
        The number of channels in the hidden layers.

    filter_channels (int):
        The number of channels in the filter layers.

    n_heads (int):
        The number of attention heads.

    n_layers (int):
        The number of layers.

    kernel_size (int):
        The size of the kernel.

    p_dropout (float):
        The dropout probability.

    resblock (str):
        The type of residual block.

    resblock_kernel_sizes (List[int]):
        The kernel sizes for the residual blocks.

    resblock_dilation_sizes (List[List[int]]):
        The dilation sizes for the residual blocks.

    upsample_rates (List[int]):
        The upsample rates.

    upsample_initial_channel (int):
        The number of channels in the initial upsample layer.

    upsample_kernel_sizes (List[int]):
        The kernel sizes for the upsample layers.

    n_layers_q (int):
        The number of layers in the quantization network.

    use_spectral_norm (bool):
        Whether to use spectral normalization.

    gin_channels (int):
        The number of channels in the global conditioning vector.

    tau (float):
        Tau parameter for the posterior encoder
    """

    zero_g: bool = field(default=True)
    inter_channels: int = field(default=192)
    hidden_channels: int = field(default=192)
    filter_channels: int = field(default=768)
    n_heads: int = field(default=2)
    n_layers: int = field(default=6)
    kernel_size: int = field(default=3)
    p_dropout: float = field(default=0.1)
    resblock: str = field(default="1")
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel: int = field(default=512)
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4])
    n_layers_q: int = field(default=3)
    use_spectral_norm: bool = field(default=False)
    gin_channels: int = field(default=256)
    tau: float = field(default=0.3)


@dataclass
class OpenVoiceConfig(BaseVCConfig):
    """Defines parameters for OpenVoice VC model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (OpenVoiceArgs):
            Model architecture arguments. Defaults to `OpenVoiceArgs()`.

        audio (OpenVoiceAudioConfig):
            Audio processing configuration. Defaults to `OpenVoiceAudioConfig()`.

        return_wav (bool):
            If true, data loader returns the waveform as well as the other outputs. Do not change. Defaults to `True`.

        compute_linear_spec (bool):
            If true, the linear spectrogram is computed and returned alongside the mel output. Do not change. Defaults to `True`.

        use_weighted_sampler (bool):
            If true, use weighted sampler with bucketing for balancing samples between datasets used in training. Defaults to `False`.

        weighted_sampler_attrs (dict):
            Key retuned by the formatter to be used for weighted sampler. For example `{"root_path": 2.0, "speaker_name": 1.0}` sets sample probabilities
            by overweighting `root_path` by 2.0. Defaults to `{}`.

        weighted_sampler_multipliers (dict):
            Weight each unique value of a key returned by the formatter for weighted sampling.
            For example `{"root_path":{"/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-100/":1.0, "/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-360/": 0.5}`.
            It will sample instances from `train-clean-100` 2 times more than `train-clean-360`. Defaults to `{}`.

        r (int):
            Number of spectrogram frames to be generated at a time. Do not change. Defaults to `1`.

        add_blank (bool):
            If true, a blank token is added in between every character. Defaults to `True`.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseVCConfig` for the inherited parameters.

    Example:

        >>> from TTS.vc.configs.openvoice_config import OpenVoiceConfig
        >>> config = OpenVoiceConfig()
    """

    model: str = "openvoice"
    # model specific params
    model_args: OpenVoiceArgs = field(default_factory=OpenVoiceArgs)
    audio: OpenVoiceAudioConfig = field(default_factory=OpenVoiceAudioConfig)

    # optimizer
    # TODO with training support

    # loss params
    # TODO with training support

    # data loader params
    return_wav: bool = True
    compute_linear_spec: bool = True

    # sampler params
    use_weighted_sampler: bool = False  # TODO: move it to the base config
    weighted_sampler_attrs: dict = field(default_factory=dict)
    weighted_sampler_multipliers: dict = field(default_factory=dict)

    # overrides
    r: int = 1  # DO NOT CHANGE
    add_blank: bool = True

    # multi-speaker settings
    # use speaker embedding layer
    num_speakers: int = 0
    speakers_file: str | None = None
    speaker_embedding_channels: int = 256

    # use d-vectors
    use_d_vector_file: bool = False
    d_vector_file: list[str] | None = None
    d_vector_dim: int | None = None

    def __post_init__(self) -> None:
        for key, val in self.model_args.items():
            if hasattr(self, key):
                self[key] = val
