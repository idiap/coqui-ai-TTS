import torch

from TTS.utils.generic_utils import is_pytorch_at_least_2_4


class SpeakerManager:
    def __init__(self, speaker_file_path=None):
        self.speakers = torch.load(speaker_file_path, weights_only=is_pytorch_at_least_2_4())

    @property
    def name_to_id(self):
        return self.speakers

    @property
    def num_speakers(self):
        return len(self.name_to_id)

    @property
    def speaker_names(self):
        return list(self.name_to_id.keys())


class LanguageManager:
    def __init__(self, config):
        self.langs = config["languages"]

    @property
    def name_to_id(self):
        return self.langs

    @property
    def num_languages(self):
        return len(self.name_to_id)

    @property
    def language_names(self):
        return list(self.name_to_id)
