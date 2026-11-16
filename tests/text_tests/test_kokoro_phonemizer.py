import pytest

from TTS.tts.utils.text.phonemizers import get_phonemizer_by_name
from TTS.tts.utils.text.phonemizers.espeak_wrapper import _is_tool
from TTS.tts.utils.text.phonemizers.kokoro_phonemizer import KokoroPhonemizer, _map_english, _map_other

requires_espeak_ng = pytest.mark.skipif(not _is_tool("espeak-ng"), reason="espeak-ng not installed")


def test_map_english():
    # Raw espeak-ng output with tie characters
    assert (
        _map_english("fˈa^ɪv bˈɔ^ɪz d^ʒˈɔ^ɪnd ðə kwˈa^ɪ^ɚ tədˈe^ɪ", british=False) == "fˈIv bˈYz ʤˈYnd ðə kwˈIəɹ tədˈA"
    )
    assert _map_english("həlˈo^ʊ ɹˈɛdɚ ɡˈɜːl wˈɔːɾɚ", british=False) == "həlˈO ɹˈɛdəɹ ɡˈɜɹl wˈɔTəɹ"
    assert _map_english("ðə wˈɛðɚɹ ɪz", british=False) == "ðə wˈɛðəɹ ɪz"
    assert _map_english("həlˈə^ʊ t^ʃˈe^əɹ", british=True) == "həlˈQ ʧˈɛːɹ"


def test_map_other():
    assert _map_other("tˈe t^ʃˈa^ɪr ˈis nˈit^ʃe") == "tˈe ʧˈIr ˈis nˈiʧe"
    assert _map_other("ˈo^ʊ d^z t^s de- flˈaɡ") == "ˈO ʣ ʦ de flˈaɡ"


def test_registered():
    phonemizer = get_phonemizer_by_name("kokoro_phonemizer", language="es")
    assert isinstance(phonemizer, KokoroPhonemizer)
    assert phonemizer.language == "es"
    assert set(phonemizer.supported_languages()) == {"en-us", "en-gb", "es", "fr-fr", "hi", "it", "pt-br"}


def test_unsupported_language():
    phonemizer = KokoroPhonemizer(use_misaki=False)
    with pytest.raises(ValueError, match="not supported"):
        phonemizer.phonemize("hello", language="de")


@requires_espeak_ng
def test_phonemize_espeak():
    phonemizer = KokoroPhonemizer(use_misaki=False)
    assert phonemizer.phonemize("Hello, five boys!") == "həlˈO, fˈIv bˈYz!"
    assert phonemizer.phonemize("Hello, five boys!", language="en-gb") == "həlˈQ, fˈIv bˈYz!"
    assert phonemizer.phonemize("Ciao, come stai?", language="it") == "ʧˈao, kˌome stˈaj?"
    # Unsupported punctuation is mapped or removed
    assert phonemizer.phonemize("¿Qué? «Bien»", language="es") == "kˈe? “bjˈen”"
