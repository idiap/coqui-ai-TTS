PYTORCH_IMPORT_ERROR = """
Coqui TTS requires the PyTorch and Torchaudio libraries but they were not found in your
environment. Check out the instructions on the installation page:
https://pytorch.org/get-started/locally/
and follow the ones that match your environment.
"""

TORCHCODEC_IMPORT_ERROR = """
From Pytorch 2.9, the torchcodec library is required for audio IO, but it was not found in your environment. You can install it with Coqui's `codec` extra:
```
pip install coqui-tts[codec]
```
"""
