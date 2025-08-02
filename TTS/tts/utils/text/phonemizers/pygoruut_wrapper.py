import importlib
from typing import List

import pygoruut.pygoruut
from pygoruut.pygoruut import Pygoruut, PygoruutLanguages

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.punctuation import Punctuation

class Pygoruut(BasePhonemizer):
    """Pygoruut (goruut) wrapper for G2P

    Args:
        language (str):
            Valid language code for the used backend.

        version (str):
            Version of the backend to initialize. Defaults to the latest version.

        punctuations (str):
            Unused.

        keep_puncs (bool):
            Unsupported.

        use_espeak_phonemes (bool):
            Currently unsupported.

        keep_stress (bool):
            Unsupported. Currently there is stress.

    Example:

        >>> from TTS.tts.utils.text.phonemizers.pygoruut_wrapper import Pygoruut
        >>> phonemizer = Pygoruut('en')
        >>> phonemizer.phonemize("Be a voice, not an! echo?", separator="|")
        'bə| ə| voʊikɪ| noʊt| ən| ɪtʃoʊ'
    """

    def __init__(
        self,
        language: str,
        version: str = None,
        punctuations=None,
        keep_puncs=True,
        use_espeak_phonemes=False,
        keep_stress=False,
    ):
        super().__init__(language=language)
        if punctuations is None:
            self.punctuations = "\x00"
        else:
            self.punctuations = punctuations
        self.pygoruut = pygoruut.pygoruut.Pygoruut(version=version, writeable_bin_dir='')

    @staticmethod
    def name():
        return "pygoruut"
    
    def phonemize_goruut(self, text: str, separator: str = "|", tie=False, language=None) -> str:  # pylint: disable=unused-argument
        """Convert input text to phonemes.

        Goruut phonemizes the given `str` by seperating each phonetic word with `separator`.

        It doesn't affect 🐸TTS since it individually converts each character to token IDs and individual phones aren't needed.

        Examples::
            "hello how are you today?" -> `hˈɛlloʊ| hoʊw| əɹˈɛ| iˈaʊ| toʊdˈæi`

        Args:
            text (str):
                Text to be converted to phonemes.

            tie (bool, optional) : Unsupported. Default to False.
        """
        if language is None:
            language = self.language
        resp = self.pygoruut.phonemize(language=language, sentence=text)
        ph_words = []
        for word in resp.Words:
            ph_words.append(word.Phonetic)

        ph = f"{separator} ".join(ph_words)
        return ph

    def _phonemize(self, text, separator, language=None):
        return self.phonemize_goruut(text, separator, tie=False, language=language)

    def is_supported_language(self, language=None):
        """Returns True if `language` is supported by the backend"""
        if language is None:
            language = self.language
        ret = language in PygoruutLanguages().get_all_supported_languages()
        #print(language, ret)
        return ret

    @staticmethod
    def supported_languages() -> List:
        """Get a dictionary of supported languages.

        Returns:
            List: List of language codes.
        """
        return PygoruutLanguages().get_supported_languages()

    def version(self):
        """Get the version of the used backend.

        Returns:
            str: Version of the used backend.
        """
        return self.pygoruut.exact_version()

    @classmethod
    def is_available(cls):
        """Return true"""
        return True

    @property
    def version_after_name(self):
        """Returns colon plus version that should be after name, otherwise empty string if phonemizer versioning is not in use"""
        return ":" + self.version()

if __name__ == "__main__":
    e = Pygoruut(language="en")
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())

    e = Pygoruut(language="en", keep_puncs=False)
    print("`" + e.phonemize("hello how are you today?") + "`")

    e = Pygoruut(language="en", keep_puncs=True)
    print("`" + e.phonemize("hello how, are you today?") + "`")
