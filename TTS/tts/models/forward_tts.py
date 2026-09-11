import logging

import torch
from coqpit import Coqpit
from monotonic_alignment_search import maximum_path
from torch import nn

from TTS.tts.configs.forward_tts_config import ForwardTTSArgs
from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.layers.generic.aligner import AlignmentNetwork
from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import average_over_durations, expand_encoder_outputs, generate_attention, sequence_mask

logger = logging.getLogger(__name__)


class ForwardTTS(BaseTTS):
    """General forward TTS model implementation that uses an encoder-decoder architecture with an optional alignment
    network and a pitch predictor.

    If the alignment network is used, the model learns the text-to-speech alignment
    from the data instead of using pre-computed durations.

    If the pitch predictor is used, the model trains a pitch predictor that predicts average pitch value for each
    input character as in the FastPitch model.

    `ForwardTTS` can be configured to one of these architectures,

        - FastPitch
        - SpeedySpeech
        - FastSpeech
        - FastSpeech2 (requires average speech energy predictor)

    Args:
        config: Model coqpit class.

    Examples:
        >>> from TTS.tts.configs.fast_pitch_config import FastPitchConfig
        >>> from TTS.tts.models.fast_pitch import ForwardTTS
        >>> config = FastPitchConfig()
        >>> model = ForwardTTS(config)
    """

    args: ForwardTTSArgs

    def __init__(
        self,
        config: Coqpit,
        ap: None = None,
        tokenizer: None = None,
        speaker_manager: None = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)

        self.init_multispeaker()

        self.max_duration = self.args.max_duration
        self.use_aligner = self.args.use_aligner
        self.use_pitch = self.args.use_pitch
        self.use_energy = self.args.use_energy
        self.binary_loss_weight = 0.0

        self.length_scale = (
            float(self.args.length_scale) if isinstance(self.args.length_scale, int) else self.args.length_scale
        )

        self.emb = nn.Embedding(self.args.num_chars, self.args.hidden_channels)

        self.encoder = Encoder(
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.encoder_type,
            self.args.encoder_params,
            self.embedded_speaker_dim,
        )

        if self.args.positional_encoding:
            self.pos_encoder = PositionalEncoding(self.args.hidden_channels)

        self.decoder = Decoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.decoder_type,
            self.args.decoder_params,
        )

        self.duration_predictor = DurationPredictor(
            self.args.hidden_channels,
            self.args.duration_predictor_hidden_channels,
            self.args.duration_predictor_kernel_size,
            self.args.duration_predictor_dropout_p,
        )

        if self.args.use_pitch:
            self.pitch_predictor = DurationPredictor(
                self.args.hidden_channels,
                self.args.pitch_predictor_hidden_channels,
                self.args.pitch_predictor_kernel_size,
                self.args.pitch_predictor_dropout_p,
            )
            self.pitch_emb = nn.Conv1d(
                1,
                self.args.hidden_channels,
                kernel_size=self.args.pitch_embedding_kernel_size,
                padding=int((self.args.pitch_embedding_kernel_size - 1) / 2),
            )

        if self.args.use_energy:
            self.energy_predictor = DurationPredictor(
                self.args.hidden_channels,
                self.args.energy_predictor_hidden_channels,
                self.args.energy_predictor_kernel_size,
                self.args.energy_predictor_dropout_p,
            )
            self.energy_emb = nn.Conv1d(
                1,
                self.args.hidden_channels,
                kernel_size=self.args.energy_embedding_kernel_size,
                padding=int((self.args.energy_embedding_kernel_size - 1) / 2),
            )

        if self.args.use_aligner:
            self.aligner = AlignmentNetwork(
                in_query_channels=self.args.out_channels, in_key_channels=self.args.hidden_channels
            )

    def _init_speaker_embedding(self) -> None:
        self.emb_g = nn.Embedding(self.num_speakers, self.args.hidden_channels)
        nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    def _init_d_vector(self) -> None:
        self.embedded_speaker_dim = self.config.d_vector_dim
        if self.args.d_vector_dim != self.args.hidden_channels:
            # self.proj_g = nn.Conv1d(self.args.d_vector_dim, self.args.hidden_channels, 1)
            self.proj_g = nn.Linear(in_features=self.args.d_vector_dim, out_features=self.args.hidden_channels)

    def format_durations(self, o_dr_log, x_mask):
        """Format predicted durations.
        1. Convert to linear scale from log scale
        2. Apply the length scale for speed adjustment
        3. Apply masking.
        4. Cast 0 durations to 1.
        5. Round the duration values.

        Args:
            o_dr_log: Log scale durations.
            x_mask: Input text mask.

        Shapes:
            - o_dr_log: :math:`(B, T_{de})`
            - x_mask: :math:`(B, T_{en})`
        """
        o_dr = (torch.exp(o_dr_log) - 1) * x_mask * self.length_scale
        o_dr[o_dr < 1] = 1.0
        o_dr = torch.round(o_dr)
        return o_dr

    def _forward_encoder(
        self, x: torch.LongTensor, x_mask: torch.FloatTensor, g: torch.FloatTensor | None = None
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Encoding forward pass.

        1. Embed speaker IDs if multi-speaker mode.
        2. Embed character sequences.
        3. Run the encoder network.
        4. Sum encoder outputs and speaker embeddings

        Args:
            x (torch.LongTensor): Input sequence IDs.
            x_mask (torch.FloatTensor): Input squence mask.
            g (torch.FloatTensor, optional): Conditioning vectors. In general speaker embeddings. Defaults to None.

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
                encoder output, encoder output for the duration predictor, input sequence mask, speaker embeddings,
                character embeddings

        Shapes:
            - x: :math:`(B, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - g: :math:`(B, C)`
        """
        # [B, T, C]
        x_emb = self.emb(x)
        # encoder pass
        # o_en = self.encoder(torch.transpose(x_emb, 1, -1), x_mask)
        o_en = self.encoder(torch.transpose(x_emb, 1, -1), x_mask, g)
        # speaker conditioning
        # TODO: try different ways of conditioning
        if g is not None:
            if hasattr(self, "proj_g"):
                g = self.proj_g(g.view(g.shape[0], -1)).unsqueeze(-1)
            o_en = o_en + g
        return o_en, x_mask, g, x_emb

    def _forward_decoder(
        self,
        o_en: torch.FloatTensor,
        dr: torch.IntTensor,
        x_mask: torch.FloatTensor,
        y_lengths: torch.IntTensor,
        g: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Decoding forward pass.

        1. Compute the decoder output mask
        2. Expand encoder output with the durations.
        3. Apply position encoding.
        4. Add speaker embeddings if multi-speaker mode.
        5. Run the decoder.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            dr (torch.IntTensor): Ground truth durations or alignment network durations.
            x_mask (torch.IntTensor): Input sequence mask.
            y_lengths (torch.IntTensor): Output sequence lengths.
            g (torch.FloatTensor): Conditioning vectors. In general speaker embeddings.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Decoder output, attention map from durations.
        """
        # expand o_en with durations
        o_en_ex, attn, y_mask = expand_encoder_outputs(o_en, dr, x_mask, y_lengths)
        # positional encoding
        if hasattr(self, "pos_encoder"):
            o_en_ex = self.pos_encoder(o_en_ex, y_mask)
        # decoder pass
        o_de = self.decoder(o_en_ex, y_mask, g=g)
        return o_de.transpose(1, 2), attn.transpose(1, 2)

    def _forward_pitch_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        pitch: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Pitch predictor forward pass.

        1. Predict pitch from encoder outputs.
        2. In training - Compute average pitch values for each input character from the ground truth pitch values.
        3. Embed average pitch values.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            x_mask (torch.IntTensor): Input sequence mask.
            pitch (torch.FloatTensor, optional): Ground truth pitch values. Defaults to None.
            dr (torch.IntTensor, optional): Ground truth durations. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Pitch embedding, pitch prediction.

        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """
        o_pitch = self.pitch_predictor(o_en, x_mask)
        if pitch is not None:
            avg_pitch = average_over_durations(pitch, dr)
            o_pitch_emb = self.pitch_emb(avg_pitch)
            return o_pitch_emb, o_pitch, avg_pitch
        o_pitch_emb = self.pitch_emb(o_pitch)
        return o_pitch_emb, o_pitch

    def _forward_energy_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        energy: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Energy predictor forward pass.

        1. Predict energy from encoder outputs.
        2. In training - Compute average pitch values for each input character from the ground truth pitch values.
        3. Embed average energy values.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            x_mask (torch.IntTensor): Input sequence mask.
            energy (torch.FloatTensor, optional): Ground truth energy values. Defaults to None.
            dr (torch.IntTensor, optional): Ground truth durations. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Energy embedding, energy prediction.

        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """
        o_energy = self.energy_predictor(o_en, x_mask)
        if energy is not None:
            avg_energy = average_over_durations(energy, dr)
            o_energy_emb = self.energy_emb(avg_energy)
            return o_energy_emb, o_energy, avg_energy
        o_energy_emb = self.energy_emb(o_energy)
        return o_energy_emb, o_energy

    def _forward_aligner(
        self, x: torch.FloatTensor, y: torch.FloatTensor, x_mask: torch.IntTensor, y_mask: torch.IntTensor
    ) -> tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Aligner forward pass.

        1. Compute a mask to apply to the attention map.
        2. Run the alignment network.
        3. Apply MAS to compute the hard alignment map.
        4. Compute the durations from the hard alignment map.

        Args:
            x (torch.FloatTensor): Input sequence.
            y (torch.FloatTensor): Output sequence.
            x_mask (torch.IntTensor): Input sequence mask.
            y_mask (torch.IntTensor): Output sequence mask.

        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                Durations from the hard alignment map, soft alignment potentials, log scale alignment potentials,
                hard alignment map.

        Shapes:
            - x: :math:`[B, T_en, C_en]`
            - y: :math:`[B, T_de, C_de]`
            - x_mask: :math:`[B, 1, T_en]`
            - y_mask: :math:`[B, 1, T_de]`

            - o_alignment_dur: :math:`[B, T_en]`
            - alignment_soft: :math:`[B, T_en, T_de]`
            - alignment_logprob: :math:`[B, 1, T_de, T_en]`
            - alignment_mas: :math:`[B, T_en, T_de]`
        """
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        alignment_soft, alignment_logprob = self.aligner(y.transpose(1, 2), x.transpose(1, 2), x_mask, None)
        alignment_mas = maximum_path(
            alignment_soft.squeeze(1).transpose(1, 2).contiguous(), attn_mask.squeeze(1).contiguous()
        )
        o_alignment_dur = torch.sum(alignment_mas, -1).int()
        alignment_soft = alignment_soft.squeeze(1).transpose(1, 2)
        return o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        y_lengths: torch.LongTensor,
        y: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
        pitch: torch.FloatTensor = None,
        energy: torch.FloatTensor = None,
        aux_input: dict = {"d_vectors": None, "speaker_ids": None},  # pylint: disable=unused-argument
    ) -> dict:
        """Model's forward pass.

        Args:
            x (torch.LongTensor): Input character sequences.
            x_lengths (torch.LongTensor): Input sequence lengths.
            y_lengths (torch.LongTensor): Output sequnce lengths. Defaults to None.
            y (torch.FloatTensor): Spectrogram frames. Only used when the alignment network is on. Defaults to None.
            dr (torch.IntTensor): Character durations over the spectrogram frames. Only used when the alignment network is off. Defaults to None.
            pitch (torch.FloatTensor): Pitch values for each spectrogram frame. Only used when the pitch predictor is on. Defaults to None.
            energy (torch.FloatTensor): energy values for each spectrogram frame. Only used when the energy predictor is on. Defaults to None.
            aux_input (Dict): Auxiliary model inputs for multi-speaker training. Defaults to `{"d_vectors": 0, "speaker_ids": None}`.

        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - y_lengths: :math:`[B]`
            - y: :math:`[B, T_max2]`
            - dr: :math:`[B, T_max]`
            - g: :math:`[B, C]`
            - pitch: :math:`[B, 1, T]`
        """
        g = self._get_speaker_conditioning(aux_input, "emb_g", normalize_d_vector=False, normalize_embedding=False)
        # compute sequence masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).float()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).float()
        # encoder pass
        o_en, x_mask, g, x_emb = self._forward_encoder(x, x_mask, g)
        # duration predictor pass
        if self.args.detach_duration_predictor:
            o_dr_log = self.duration_predictor(o_en.detach(), x_mask)
        else:
            o_dr_log = self.duration_predictor(o_en, x_mask)
        o_dr = torch.clamp(torch.exp(o_dr_log) - 1, 0, self.max_duration)
        # generate attn mask from predicted durations
        o_attn = generate_attention(o_dr.squeeze(1), x_mask)
        # aligner
        o_alignment_dur = None
        alignment_soft = None
        alignment_logprob = None
        alignment_mas = None
        if self.use_aligner:
            o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas = self._forward_aligner(
                x_emb, y, x_mask, y_mask
            )
            alignment_soft = alignment_soft.transpose(1, 2)
            alignment_mas = alignment_mas.transpose(1, 2)
            dr = o_alignment_dur
        # pitch predictor pass
        o_pitch = None
        avg_pitch = None
        if self.args.use_pitch:
            o_pitch_emb, o_pitch, avg_pitch = self._forward_pitch_predictor(o_en, x_mask, pitch, dr)
            o_en = o_en + o_pitch_emb
        # energy predictor pass
        o_energy = None
        avg_energy = None
        if self.args.use_energy:
            o_energy_emb, o_energy, avg_energy = self._forward_energy_predictor(o_en, x_mask, energy, dr)
            o_en = o_en + o_energy_emb
        # decoder pass
        o_de, attn = self._forward_decoder(
            o_en, dr, x_mask, y_lengths, g=None
        )  # TODO: maybe pass speaker embedding (g) too
        outputs = {
            "model_outputs": o_de,  # [B, T, C]
            "durations_log": o_dr_log.squeeze(1),  # [B, T]
            "durations": o_dr.squeeze(1),  # [B, T]
            "attn_durations": o_attn,  # for visualization [B, T_en, T_de']
            "pitch_avg": o_pitch,
            "pitch_avg_gt": avg_pitch,
            "energy_avg": o_energy,
            "energy_avg_gt": avg_energy,
            "alignments": attn,  # [B, T_de, T_en]
            "alignment_soft": alignment_soft,
            "alignment_mas": alignment_mas,
            "o_alignment_dur": o_alignment_dur,
            "alignment_logprob": alignment_logprob,
            "x_mask": x_mask,
            "y_mask": y_mask,
        }
        return outputs

    @torch.inference_mode()
    def inference(self, x, aux_input={"d_vectors": None, "speaker_ids": None}):  # pylint: disable=unused-argument
        """Model's inference pass.

        Args:
            x (torch.LongTensor): Input character sequence.
            aux_input (Dict): Auxiliary model inputs. Defaults to `{"d_vectors": None, "speaker_ids": None}`.

        Shapes:
            - x: [B, T_max]
            - x_lengths: [B]
            - g: [B, C]
        """
        g = self._get_speaker_conditioning(aux_input, "emb_g", normalize_d_vector=False, normalize_embedding=False)
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).to(x.dtype).float()
        # encoder pass
        o_en, x_mask, g, _ = self._forward_encoder(x, x_mask, g)
        # duration predictor pass
        o_dr_log = self.duration_predictor(o_en.squeeze(), x_mask)
        o_dr = self.format_durations(o_dr_log, x_mask).squeeze(1)
        y_lengths = o_dr.sum(1)

        # pitch predictor pass
        o_pitch = None
        if self.args.use_pitch:
            o_pitch_emb, o_pitch = self._forward_pitch_predictor(o_en, x_mask)
            o_en = o_en + o_pitch_emb
        # energy predictor pass
        o_energy = None
        if self.args.use_energy:
            o_energy_emb, o_energy = self._forward_energy_predictor(o_en, x_mask)
            o_en = o_en + o_energy_emb
        # decoder pass
        o_de, attn = self._forward_decoder(o_en, o_dr, x_mask, y_lengths, g=None)
        outputs = {
            "model_outputs": o_de,
            "alignments": attn,
            "pitch": o_pitch,
            "energy": o_energy,
            "durations_log": o_dr_log,
        }
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        pitch = batch["pitch"] if self.args.use_pitch else None
        energy = batch["energy"] if self.args.use_energy else None
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        durations = batch["durations"]
        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}

        # forward pass
        outputs = self.forward(
            text_input,
            text_lengths,
            mel_lengths,
            y=mel_input,
            dr=durations,
            pitch=pitch,
            energy=energy,
            aux_input=aux_input,
        )
        # use aligner's output as the duration target
        if self.use_aligner:
            durations = outputs["o_alignment_dur"]
        # use float32 in AMP
        with torch.autocast("cuda", enabled=False):
            # compute loss
            loss_dict = criterion(
                decoder_output=outputs["model_outputs"],
                decoder_target=mel_input,
                decoder_output_lens=mel_lengths,
                dur_output=outputs["durations_log"],
                dur_target=durations,
                pitch_output=outputs["pitch_avg"] if self.use_pitch else None,
                pitch_target=outputs["pitch_avg_gt"] if self.use_pitch else None,
                energy_output=outputs["energy_avg"] if self.use_energy else None,
                energy_target=outputs["energy_avg_gt"] if self.use_energy else None,
                input_lens=text_lengths,
                alignment_logprob=outputs["alignment_logprob"] if self.use_aligner else None,
                alignment_soft=outputs["alignment_soft"],
                alignment_hard=outputs["alignment_mas"],
                binary_loss_weight=self.binary_loss_weight,
            )
            # compute duration error
            durations_pred = outputs["durations"]
            duration_error = torch.abs(durations - durations_pred).sum() / text_lengths.sum()
            loss_dict["duration_error"] = duration_error

        return outputs, loss_dict

    def _create_logs(self, batch, outputs):
        """Create common logger outputs."""
        from TTS.tts.utils.visual import plot_alignment, plot_avg_energy, plot_avg_pitch, plot_spectrogram

        model_outputs = outputs["model_outputs"]
        alignments = outputs["alignments"]
        mel_input = batch["mel_input"]

        pred_spec = model_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, self.ap, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, self.ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        # plot pitch figures
        if self.args.use_pitch:
            pitch_avg = abs(outputs["pitch_avg_gt"][0, 0].data.cpu().numpy())
            pitch_avg_hat = abs(outputs["pitch_avg"][0, 0].data.cpu().numpy())
            chars = self.tokenizer.decode(batch["text_input"][0].data.cpu().numpy())
            pitch_figures = {
                "pitch_ground_truth": plot_avg_pitch(pitch_avg, chars, output_fig=False),
                "pitch_avg_predicted": plot_avg_pitch(pitch_avg_hat, chars, output_fig=False),
            }
            figures.update(pitch_figures)

        # plot energy figures
        if self.args.use_energy:
            energy_avg = abs(outputs["energy_avg_gt"][0, 0].data.cpu().numpy())
            energy_avg_hat = abs(outputs["energy_avg"][0, 0].data.cpu().numpy())
            chars = self.tokenizer.decode(batch["text_input"][0].data.cpu().numpy())
            energy_figures = {
                "energy_ground_truth": plot_avg_energy(energy_avg, chars, output_fig=False),
                "energy_avg_predicted": plot_avg_energy(energy_avg_hat, chars, output_fig=False),
            }
            figures.update(energy_figures)

        # plot the attention mask computed from the predicted durations
        if "attn_durations" in outputs:
            alignments_hat = outputs["attn_durations"][0].data.cpu().numpy()
            figures["alignment_hat"] = plot_alignment(alignments_hat.T, output_fig=False)

        # Sample audio
        train_audio = self.ap.inv_melspectrogram(pred_spec.T)
        return figures, {"audio": train_audio}

    def get_criterion(self):
        from TTS.tts.layers.losses import ForwardTTSLoss  # pylint: disable=import-outside-toplevel

        return ForwardTTSLoss(self.config)

    def on_train_step_start(self, trainer):
        """Schedule binary loss weight."""
        self.binary_loss_weight = min(trainer.epochs_done / self.config.binary_loss_warmup_epochs, 1.0) * 1.0
