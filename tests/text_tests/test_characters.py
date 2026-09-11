import pytest

from TTS.tts.utils.text.characters import BaseCharacters, BaseVocabulary, Graphemes, IPAPhonemes


class TestBaseVocabulary:
    @pytest.fixture
    def phonemes(self):
        return IPAPhonemes()

    @pytest.fixture
    def base_vocab(self, phonemes):
        return BaseVocabulary(
            vocab=phonemes._vocab,
            pad=phonemes.pad,
            blank=phonemes.blank,
            bos=phonemes.bos,
            eos=phonemes.eos,
        )

    @pytest.fixture
    def empty_vocab(self):
        return BaseVocabulary({})

    def test_pad_id(self, empty_vocab, base_vocab, phonemes):
        assert empty_vocab.pad_id == 0
        assert base_vocab.pad_id == phonemes.pad_id

    def test_blank_id(self, empty_vocab, base_vocab, phonemes):
        assert empty_vocab.blank_id == 0
        assert base_vocab.blank_id == phonemes.blank_id

    def test_vocab(self, empty_vocab, base_vocab, phonemes):
        assert empty_vocab.vocab == {}
        assert base_vocab.vocab == phonemes._vocab

    def test_num_chars(self, empty_vocab, base_vocab, phonemes):
        assert empty_vocab.num_chars == 0
        assert base_vocab.num_chars == phonemes.num_chars

    def test_char_to_id(self, empty_vocab, base_vocab, phonemes):
        with pytest.raises(KeyError):
            empty_vocab.char_to_id("a")
        for k in phonemes.vocab:
            assert base_vocab.char_to_id(k) == phonemes.char_to_id(k)

    def test_id_to_char(self, empty_vocab, base_vocab, phonemes):
        with pytest.raises(KeyError):
            empty_vocab.id_to_char(0)
        for k in phonemes.vocab:
            v = phonemes.char_to_id(k)
            assert base_vocab.id_to_char(v) == phonemes.id_to_char(v)


class TestBaseCharacters:
    def test_default_character_sets(self):
        """Test initiation of default character sets"""
        IPAPhonemes()
        Graphemes()

    def test_unique(self):
        """Test if the unique option works"""
        characters = BaseCharacters(
            characters="bacc",
            punctuations=".,;:!? ",
            pad="[PAD]",
            eos="[EOS]",
            bos="[BOS]",
            blank="[BLANK]",
            is_unique=True,
        )

        assert characters.num_chars == len(
            ["[PAD]", "[EOS]", "[BOS]", "[BLANK]", "b", "a", "c", ".", ",", ";", ":", "!", "?", " "]
        )

    def test_unique_sorted(self):
        """Test if the unique and sorted option works"""
        characters = BaseCharacters(
            characters="ccba",
            punctuations=".,;:!? ",
            pad="[PAD]",
            eos="[EOS]",
            bos="[BOS]",
            blank="[BLANK]",
            is_unique=True,
            is_sorted=True,
        )

        assert characters.num_chars == len(
            ["[PAD]", "[EOS]", "[BOS]", "[BLANK]", "a", "b", "c", ".", ",", ";", ":", "!", "?", " "]
        )

    def test_getters(self):
        """Test the class getters behave as expected"""
        characters = BaseCharacters(
            characters="abc",
            punctuations=".,;:!? ",
            pad="[PAD]",
            eos="[EOS]",
            bos="[BOS]",
            blank="[BLANK]",
            is_unique=True,
        )
        assert characters.characters == "abc"
        assert characters.punctuations == ".,;:!? "
        assert characters.pad == "[PAD]"
        assert characters.eos == "[EOS]"
        assert characters.bos == "[BOS]"
        assert characters.blank == "[BLANK]"
        assert characters.vocab == [
            "[PAD]",
            "[EOS]",
            "[BOS]",
            "[BLANK]",
            "a",
            "b",
            "c",
            ".",
            ",",
            ";",
            ":",
            "!",
            "?",
            " ",
        ]
        assert characters.num_chars == len(
            ["[PAD]", "[EOS]", "[BOS]", "[BLANK]", "a", "b", "c", ".", ",", ";", ":", "!", "?", " "]
        )

        characters.print_log()

    def test_char_lookup(self):
        """Test char to ID and ID to char conversion"""
        characters = BaseCharacters(
            characters="abc",
            punctuations=".,;:!? ",
            pad="[PAD]",
            eos="[EOS]",
            bos="[BOS]",
            blank="[BLANK]",
            is_unique=True,
        )

        # char to ID
        assert characters.char_to_id("[PAD]") == 0
        assert characters.char_to_id("[EOS]") == 1
        assert characters.char_to_id("[BOS]") == 2
        assert characters.char_to_id("[BLANK]") == 3
        assert characters.char_to_id("a") == 4
        assert characters.char_to_id("b") == 5
        assert characters.char_to_id("c") == 6
        assert characters.char_to_id(".") == 7
        assert characters.char_to_id(",") == 8
        assert characters.char_to_id(";") == 9
        assert characters.char_to_id(":") == 10
        assert characters.char_to_id("!") == 11
        assert characters.char_to_id("?") == 12
        assert characters.char_to_id(" ") == 13

        # ID to char
        assert characters.id_to_char(0) == "[PAD]"
        assert characters.id_to_char(1) == "[EOS]"
        assert characters.id_to_char(2) == "[BOS]"
        assert characters.id_to_char(3) == "[BLANK]"
        assert characters.id_to_char(4) == "a"
        assert characters.id_to_char(5) == "b"
        assert characters.id_to_char(6) == "c"
        assert characters.id_to_char(7) == "."
        assert characters.id_to_char(8) == ","
        assert characters.id_to_char(9) == ";"
        assert characters.id_to_char(10) == ":"
        assert characters.id_to_char(11) == "!"
        assert characters.id_to_char(12) == "?"
        assert characters.id_to_char(13) == " "
