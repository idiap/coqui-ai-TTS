"""Decoder for StyleTTS 2."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
from torch.nn.utils import remove_weight_norm, weight_norm

from TTS.tts.configs.styletts2_config import StyleTTS2DecoderConfig
from TTS.tts.layers.styletts2.common import AdaIN1d, AdaINResBlk1d
from TTS.vc.layers.freevc.commons import init_weights
from TTS.vocoder.models.hifigan_generator import LRELU_SLOPE, get_padding


class AdaINResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
            ]
        )
        self.convs2.apply(init_weights)

        self.adain1 = nn.ModuleList(
            [
                AdaIN1d(style_dim, channels),
                AdaIN1d(style_dim, channels),
                AdaIN1d(style_dim, channels),
            ]
        )

        self.adain2 = nn.ModuleList(
            [
                AdaIN1d(style_dim, channels),
                AdaIN1d(style_dim, channels),
                AdaIN1d(style_dim, channels),
            ]
        )

        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class HifiGANGenerator(nn.Module):
    """HifiGAN-based waveform generator for StyleTTS 2."""

    def __init__(self, config: StyleTTS2DecoderConfig, hidden_channels: int, style_dim: int = 128):
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000, upsample_scale=np.prod(config.upsample_rates), harmonic_num=8, voiced_threshod=10
        )

        self.f0_upsamp = nn.Upsample(scale_factor=int(np.prod(config.upsample_rates)))
        self.noise_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.noise_res = nn.ModuleList()

        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            c_cur = config.upsample_initial_channel // (2 ** (i + 1))

            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        config.upsample_initial_channel // (2**i),
                        config.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(u // 2 + u % 2),
                        output_padding=u % 2,
                    )
                )
            )

            if i + 1 < len(config.upsample_rates):
                stride_f0 = int(np.prod(config.upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    nn.Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0 + 1) // 2)
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim))

        self.resblocks = nn.ModuleList()

        self.alphas = nn.ParameterList()
        self.alphas.append(nn.Parameter(torch.ones(1, config.upsample_initial_channel, 1)))

        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))

            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x: torch.Tensor, style: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t

        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2)

        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x_source = self.noise_convs[i](har_source)
            x_source = self.noise_res[i](x_source, style)

            x = self.ups[i](x)
            x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, style)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, style)
            x = xs / self.num_kernels
        x = x + (1 / self.alphas[i + 1]) * (torch.sin(self.alphas[i + 1] * x) ** 2)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class TorchSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window="hann"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(input_data.device),
            return_complex=True,
        )

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
        )

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class SineGen(nn.Module):
    """Definition of sine generator.

    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self,
        samp_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The integer part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # # for normal case

            # # To prevent torch.cumsum numerical overflow,
            # # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # # Buffer tmp_over_one_idx indicates the time step to add -1.
            # # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            # tmp_over_one = torch.cumsum(rad_values, 1) % 1
            # tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
            # cumsum_shift = torch.zeros_like(rad_values)
            # cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            # phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            rad_values = F.interpolate(
                rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear"
            ).transpose(1, 2)

            # tmp_over_one = torch.cumsum(rad_values, 1) % 1
            # tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
            # cumsum_shift = torch.zeros_like(rad_values)
            # cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = F.interpolate(
                phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear"
            ).transpose(1, 2)
            sines = torch.sin(phase)

        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0
    ):
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class iSTFTNetGenerator(nn.Module):
    """iSTFTNet-based waveform generator for StyleTTS 2."""

    def __init__(self, config: StyleTTS2DecoderConfig, hidden_channels: int, style_dim: int = 128):
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=np.prod(config.upsample_rates) * config.gen_istft_hop_size,
            harmonic_num=8,
            voiced_threshod=10,
        )
        self.f0_upsamp = nn.Upsample(scale_factor=int(np.prod(config.upsample_rates) * config.gen_istft_hop_size))
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        config.upsample_initial_channel // (2**i),
                        config.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))

            c_cur = config.upsample_initial_channel // (2 ** (i + 1))

            if i + 1 < len(config.upsample_rates):
                stride_f0 = int(np.prod(config.upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    nn.Conv1d(
                        config.gen_istft_n_fft + 2,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(config.gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim))

        self.post_n_fft = config.gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            filter_length=config.gen_istft_n_fft,
            hop_length=config.gen_istft_hop_size,
            win_length=config.gen_istft_n_fft,
        )

    def forward(self, x: torch.Tensor, style: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source, _, _ = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, style)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, style)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, style)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        result = self.stft.inverse(spec, phase)
        return result

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class StyleTTSDecoder(nn.Module):
    """StyleTTS 2 decoder that can use either HifiGAN or iSTFTNet waveform generator."""

    def __init__(
        self,
        config: StyleTTS2DecoderConfig,
        hidden_channels: int = 512,
        style_dim: int = 128,
    ):
        super().__init__()

        self.encode = AdaINResBlk1d(hidden_channels + 2, 1024, style_dim)
        self.decode = nn.ModuleList(
            [
                AdaINResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdaINResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdaINResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdaINResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
            ]
        )

        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.energy_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.text_residual = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))

        generator_class = HifiGANGenerator if config.generator_type == "hifigan" else iSTFTNetGenerator
        self.generator = generator_class(config, hidden_channels, style_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        style: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass - modified to match original StyleTTS2 decoder.

        Args:
            text_features: Text features [B, hidden_channels, T_text]
            style: Style vector [B, style_dim]
            f0: F0 values [B, T_audio] (optional)
            energy: Energy values [B, T_audio] (optional)

        Returns:
            Dictionary containing generated waveform and features
        """
        if self.training:
            raise NotImplementedError

        f0_conv = self.F0_conv(f0.unsqueeze(1))
        energy_conv = self.energy_conv(energy.unsqueeze(1))

        x = torch.cat([text_features, f0_conv, energy_conv], dim=1)
        x = self.encode(x, style)

        text = self.text_residual(text_features)

        residual = True
        for block in self.decode:
            if residual:
                x = torch.cat([x, text, f0_conv, energy_conv], dim=1)
            x = block(x, style)
            if block.do_upsample:
                residual = False

        return self.generator(x, style, f0)
