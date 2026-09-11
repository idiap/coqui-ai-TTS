import numpy as np
import torch

from TTS.utils.audio import numpy_transforms as np_transforms
from TTS.utils.audio.torch_transforms import amp_to_db, db_to_amp


def test_amplitude_db_conversion():
    x = torch.rand(11)
    o1 = amp_to_db(x=x, spec_gain=1.0)
    o2 = db_to_amp(x=o1, spec_gain=1.0)
    np_o1 = np_transforms.amp_to_db(x=x.cpu().numpy(), base=np.e)
    np_o2 = np_transforms.db_to_amp(x=np_o1, base=np.e)
    assert torch.allclose(x, o2)
    assert torch.allclose(o1, torch.tensor(np_o1))
    assert torch.allclose(o2, torch.tensor(np_o2))
