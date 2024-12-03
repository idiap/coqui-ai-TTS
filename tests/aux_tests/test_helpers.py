import torch as T

from TTS.tts.utils.helpers import (
    average_over_durations,
    expand_encoder_outputs,
    generate_attention,
    generate_path,
    rand_segments,
    segment,
    sequence_mask,
)


def test_average_over_durations():  # pylint: disable=no-self-use
    pitch = T.rand(1, 1, 128)

    durations = T.randint(1, 5, (1, 21))
    coeff = 128.0 / durations.sum()
    durations = T.floor(durations * coeff)
    diff = 128.0 - durations.sum()
    durations[0, -1] += diff
    durations = durations.long()

    pitch_avg = average_over_durations(pitch, durations)

    index = 0
    for idx, dur in enumerate(durations[0]):
        assert abs(pitch_avg[0, 0, idx] - pitch[0, 0, index : index + dur.item()].mean()) < 1e-5
        index += dur


def test_sequence_mask():
    lengths = T.randint(10, 15, (8,))
    mask = sequence_mask(lengths)
    for i in range(8):
        l = lengths[i].item()
        assert mask[i, :l].sum() == l
        assert mask[i, l:].sum() == 0


def test_segment():
    x = T.arange(0, 12)
    x = x.repeat(8, 1).unsqueeze(1)
    segment_ids = T.randint(0, 7, (8,))

    segments = segment(x, segment_ids, segment_size=4)
    for idx, start_indx in enumerate(segment_ids):
        assert x[idx, :, start_indx : start_indx + 4].sum() == segments[idx, :, :].sum()

    try:
        segments = segment(x, segment_ids, segment_size=10)
        raise Exception("Should have failed")
    except:
        pass

    segments = segment(x, segment_ids, segment_size=10, pad_short=True)
    for idx, start_indx in enumerate(segment_ids):
        assert x[idx, :, start_indx : start_indx + 10].sum() == segments[idx, :, :].sum()


def test_rand_segments():
    x = T.rand(2, 3, 4)
    x_lens = T.randint(3, 4, (2,))
    segments, seg_idxs = rand_segments(x, x_lens, segment_size=2)
    assert segments.shape == (2, 3, 2)
    assert all(seg_idxs >= 0), seg_idxs
    try:
        segments, _ = rand_segments(x, x_lens, segment_size=5)
        raise Exception("Should have failed")
    except:
        pass
    x_lens_back = x_lens.clone()
    segments, seg_idxs = rand_segments(x, x_lens.clone(), segment_size=5, pad_short=True, let_short_samples=True)
    assert segments.shape == (2, 3, 5)
    assert all(seg_idxs >= 0), seg_idxs
    assert all(x_lens_back == x_lens)


def test_generate_path():
    durations = T.randint(1, 4, (10, 21))
    x_length = T.randint(18, 22, (10,))
    x_mask = sequence_mask(x_length, max_len=21).unsqueeze(1).long()
    durations = durations * x_mask.squeeze(1)
    y_length = durations.sum(1)
    y_mask = sequence_mask(y_length).unsqueeze(1).long()
    attn_mask = (T.unsqueeze(x_mask, -1) * T.unsqueeze(y_mask, 2)).squeeze(1).long()
    print(attn_mask.shape)
    path = generate_path(durations, attn_mask)
    assert path.shape == (10, 21, durations.sum(1).max().item())
    for b in range(durations.shape[0]):
        current_idx = 0
        for t in range(durations.shape[1]):
            assert all(path[b, t, current_idx : current_idx + durations[b, t].item()] == 1.0)
            assert all(path[b, t, :current_idx] == 0.0)
            assert all(path[b, t, current_idx + durations[b, t].item() :] == 0.0)
            current_idx += durations[b, t].item()

    assert T.all(path == generate_attention(durations, x_mask, y_mask))
    assert T.all(path == generate_attention(durations, x_mask))


def test_expand_encoder_outputs():
    inputs = T.rand(2, 5, 57)
    durations = T.randint(1, 4, (2, 57))

    x_mask = T.ones(2, 1, 57)
    y_lengths = T.ones(2) * durations.sum(1).max()

    expanded, _, _ = expand_encoder_outputs(inputs, durations, x_mask, y_lengths)

    for b in range(durations.shape[0]):
        index = 0
        for idx, dur in enumerate(durations[b]):
            idx_expanded = expanded[b, :, index : index + dur.item()]
            diff = (idx_expanded - inputs[b, :, idx].repeat(int(dur)).view(idx_expanded.shape)).sum()
            assert abs(diff) < 1e-6, diff
            index += dur
