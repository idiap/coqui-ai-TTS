import abc
import logging
from typing import Any

from TTS.tts.utils.text.punctuation import _PUNC_IDX, Punctuation

logger = logging.getLogger(__name__)


class BasePhonemizer(abc.ABC):
    """Base phonemizer class

    Phonemization follows the following steps:
        1. Preprocessing:
            - remove empty lines
            - remove punctuation
            - keep track of punctuation marks

        2. Phonemization:
            - convert text to phonemes

        3. Postprocessing:
            - join phonemes
            - restore punctuation marks

    Args:
        language (str):
            Language used by the phonemizer.

        punctuations (List[str]):
            List of punctuation marks to be preserved.

        keep_puncs (bool):
            Whether to preserve punctuation marks or not.
    """

    def __init__(
        self, language: str, punctuations: str = Punctuation.default_puncs(), *, keep_puncs: bool = False
    ) -> None:
        # ensure the backend is installed on the system
        if not self.is_available():
            raise RuntimeError(f"{self.name()} not installed on your system")  # pragma: nocover

        # ensure the backend support the requested language
        self._language = self._init_language(language)

        # setup punctuation processing
        self._keep_puncs = keep_puncs
        self._punctuator = Punctuation(punctuations)

    def _init_language(self, language: str) -> str:
        """Language initialization

        This method may be overloaded in child classes (see Segments backend)

        """
        if not self.is_supported_language(language):
            raise RuntimeError(f'language "{language}" is not supported by the {self.name()} backend')
        return language

    @property
    def language(self) -> str:
        """The language code configured to be used for phonemization"""
        return self._language

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """The name of the backend"""
        ...

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Returns True if the backend is installed, False otherwise"""
        ...

    @abc.abstractmethod
    def version(self) -> str:
        """Return the backend version as a tuple (major, minor, patch)"""
        ...

    @staticmethod
    @abc.abstractmethod
    def supported_languages() -> list[str] | dict[str, Any]:
        """Return a dict of language codes -> name supported by the backend"""
        ...

    def is_supported_language(self, language: str) -> bool:
        """Returns True if `language` is supported by the backend"""
        return language in self.supported_languages()

    @abc.abstractmethod
    def _phonemize(self, text: str, separator: str) -> str:
        """The main phonemization method"""

    def _phonemize_preprocess(self, text: str) -> tuple[list[str], list[_PUNC_IDX]]:
        """Preprocess the text before phonemization

        1. remove spaces
        2. remove punctuation

        Override this if you need a different behaviour
        """
        text = text.strip()
        if self._keep_puncs:
            # a tuple (text, punctuation marks)
            return self._punctuator.strip_to_restore(text)
        return [self._punctuator.strip(text)], []

    def _phonemize_postprocess(self, phonemized: list[str], punctuations: list[_PUNC_IDX]) -> str:
        """Postprocess the raw phonemized output

        Override this if you need a different behaviour
        """
        if self._keep_puncs:
            return self._punctuator.restore(phonemized, punctuations)[0]
        return phonemized[0]

    def phonemize(self, text: str, separator: str = "|", language: str | None = None) -> str:
        """Returns the `text` phonemized for the given language

        Args:
            text (str):
                Text to be phonemized.

            separator (str):
                string separator used between phonemes. Default to '_'.

        Returns:
            (str): Phonemized text
        """
        preprocessed, punctuations = self._phonemize_preprocess(text)
        phonemized = []
        for t in preprocessed:
            p = self._phonemize(t, separator)
            phonemized.append(p)
        return self._phonemize_postprocess(phonemized, punctuations)

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        logger.info("%s| phoneme language: %s", indent, self.language)
        logger.info("%s| phoneme backend: %s", indent, self.name())
