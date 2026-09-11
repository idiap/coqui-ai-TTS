import csv
import logging
import os
import re
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

logger = logging.getLogger(__name__)


class Formatter(Protocol):
    def __call__(
        self,
        root_path: str | os.PathLike[Any],
        meta_file: str | os.PathLike[Any],
        ignored_speakers: list[str] | None,
        **kwargs,
    ) -> list[dict[str, Any]]: ...


_FORMATTER_REGISTRY: dict[str, Formatter] = {}


def register_formatter(name: str, formatter: Formatter) -> None:
    """Add a formatter function to the registry.

    Args:
        name: Name of the formatter.
        formatter: Formatter function.
    """
    if name.lower() in _FORMATTER_REGISTRY:
        msg = f"Formatter {name} already exists."
        raise ValueError(msg)
    _FORMATTER_REGISTRY[name.lower()] = formatter


########################
# DATASETS
########################


def cml_tts(root_path, meta_file, ignored_speakers=None):
    """Normalize the CML-TTS meta data file to TTS format.

    Website: https://github.com/freds0/CML-TTS-Dataset/

    Expected metadata format (pipe-delimited CSV with header)::

        wav_filename|transcript|client_id|emotion_name
        audio/speaker1/file001.wav|This is the transcript.|speaker1|neutral
        audio/speaker2/file002.wav|Another sentence here.|speaker2|happy

    Note: The ``client_id`` and ``emotion_name`` columns are optional. If missing,
    defaults to ``"default"`` speaker and ``"neutral"`` emotion.

    Audio files are located at: ``{root_path}/{wav_filename}``
    """
    filepath = os.path.join(root_path, meta_file)
    # ensure there are 4 columns for every line
    with open(filepath, encoding="utf8") as f:
        lines = f.readlines()
    num_cols = len(lines[0].split("|"))  # take the first row as reference
    for idx, line in enumerate(lines[1:]):
        if len(line.split("|")) != num_cols:
            logger.warning("Missing column in line %d -> %s", idx + 1, line.strip())
    # load metadata
    with open(Path(root_path) / meta_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        metadata = list(reader)
    assert all(x in metadata[0] for x in ["wav_filename", "transcript"])
    client_id = None if "client_id" in metadata[0] else "default"
    emotion_name = None if "emotion_name" in metadata[0] else "neutral"
    items = []
    not_found_counter = 0
    for row in metadata:
        if client_id is None and ignored_speakers is not None and row["client_id"] in ignored_speakers:
            continue
        audio_path = os.path.join(root_path, row["wav_filename"])
        if not os.path.exists(audio_path):
            not_found_counter += 1
            continue
        items.append(
            {
                "text": row["transcript"],
                "audio_file": audio_path,
                "speaker_name": client_id if client_id is not None else row["client_id"],
                "emotion_name": emotion_name if emotion_name is not None else row["emotion_name"],
                "root_path": root_path,
            }
        )
    if not_found_counter > 0:
        logger.warning("%d files not found", not_found_counter)
    return items


def coqui(root_path, meta_file, ignored_speakers=None):
    """Normalize the Coqui internal dataset format to TTS format.

    Expected metadata format (pipe-delimited CSV with header)::

        audio_file|text|speaker_name|emotion_name
        clips/speaker1/file001.wav|This is the transcript.|speaker1|neutral
        clips/speaker2/file002.wav|Another sentence here.|speaker2|happy

    Note: The ``speaker_name`` and ``emotion_name`` columns are optional. If missing,
    defaults to ``"coqui"`` speaker and ``"neutral"`` emotion.

    Audio files are located at: ``{root_path}/{audio_file}``
    """
    filepath = os.path.join(root_path, meta_file)
    # ensure there are 4 columns for every line
    with open(filepath, encoding="utf8") as f:
        lines = f.readlines()
    num_cols = len(lines[0].split("|"))  # take the first row as reference
    for idx, line in enumerate(lines[1:]):
        if len(line.split("|")) != num_cols:
            logger.warning("Missing column in line %d -> %s", idx + 1, line.strip())
    # load metadata
    with open(Path(root_path) / meta_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        metadata = list(reader)
    assert all(x in metadata[0] for x in ["audio_file", "text"])
    speaker_name = None if "speaker_name" in metadata[0] else "coqui"
    emotion_name = None if "emotion_name" in metadata[0] else "neutral"
    items = []
    not_found_counter = 0
    for row in metadata:
        if speaker_name is None and ignored_speakers is not None and row["speaker_name"] in ignored_speakers:
            continue
        audio_path = os.path.join(root_path, row["audio_file"])
        if not os.path.exists(audio_path):
            not_found_counter += 1
            continue
        items.append(
            {
                "text": row["text"],
                "audio_file": audio_path,
                "speaker_name": speaker_name if speaker_name is not None else row["speaker_name"],
                "emotion_name": emotion_name if emotion_name is not None else row["emotion_name"],
                "root_path": root_path,
            }
        )
    if not_found_counter > 0:
        logger.warning("%d files not found", not_found_counter)
    return items


def tweb(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize TWEB dataset.

    Website: https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset

    Expected metadata format (tab-delimited)::

        file001\tThis is the transcript.
        file002\tAnother sentence here.

    Note: ``.wav`` is automatically appended to the file ID.

    Audio files should be in: ``{root_path}/{filename}.wav``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "tweb"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("\t")
            wav_file = os.path.join(root_path, cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def mozilla(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize Mozilla meta data files to TTS format.

    Expected metadata format (pipe-delimited)::

        This is the transcript.|file001.wav
        Another sentence here.|file002.wav

    Note: Text comes before the filename in this format.

    Audio files should be in: ``{root_path}/wavs/{filename}``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = cols[1].strip()
            text = cols[0].strip()
            wav_file = os.path.join(root_path, "wavs", wav_file)
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def mozilla_de(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize Mozilla German dataset meta data files to TTS format.

    Expected metadata format (pipe-delimited, ISO 8859-1 encoding)::

        1_0001.wav|This is the German transcript.
        1_0002.wav|Another sentence here.
        2_0001.wav|From a different batch.

    Note: Files are organized in batch folders named ``BATCH_{N}_FINAL`` where N
    is extracted from the filename prefix (e.g., ``1_0001.wav`` → ``BATCH_1_FINAL``).

    Audio files should be in: ``{root_path}/BATCH_{N}_FINAL/{filename}``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, encoding="ISO 8859-1") as ttf:
        for line in ttf:
            cols = line.strip().split("|")
            wav_file = cols[0].strip()
            text = cols[1].strip()
            folder_name = f"BATCH_{wav_file.split('_')[0]}_FINAL"
            wav_file = os.path.join(root_path, folder_name, wav_file)
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def mailabs(root_path, meta_files=None, ignored_speakers=None):
    """Normalize M-AI-Labs meta data files to TTS format.

    Website: https://github.com/imdatceleste/m-ailabs-dataset

    Expected metadata format (pipe-delimited)::

        file001|This is the transcript.
        file002|Another sentence here.

    Note: This dataset automatically searches for all ``metadata.csv`` files recursively
    in ``root_path`` unless ``meta_files`` is specified. Speaker names are extracted from
    the folder structure: ``by_book/{gender}/{speaker_name}/...``

    Audio files should be in: ``{folder}/wavs/{filename}.wav``

    Args:
        root_path (str): root folder of the MAILAB language folder.
        meta_files (str):  list of meta files to be used in the training. If None, finds all the csv files
            recursively. Defaults to None
    """
    speaker_regex = re.compile(f"by_book{os.sep}(male|female){os.sep}(?P<speaker_name>[^{os.sep}]+){os.sep}")
    if not meta_files:
        csv_files = glob(root_path + f"{os.sep}**{os.sep}metadata.csv", recursive=True)
    else:
        csv_files = meta_files

    # meta_files = [f.strip() for f in meta_files.split(",")]
    items = []
    for csv_file in csv_files:
        if os.path.isfile(csv_file):
            txt_file = csv_file
        else:
            txt_file = os.path.join(root_path, csv_file)

        folder = os.path.dirname(txt_file)
        # determine speaker based on folder structure...
        speaker_name_match = speaker_regex.search(txt_file)
        if speaker_name_match is None:
            continue
        speaker_name = speaker_name_match.group("speaker_name")
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_name in ignored_speakers:
                continue
        logger.info(csv_file)
        with open(txt_file, encoding="utf-8") as ttf:
            for line in ttf:
                cols = line.split("|")
                if not meta_files:
                    wav_file = os.path.join(folder, "wavs", cols[0] + ".wav")
                else:
                    wav_file = os.path.join(root_path, folder.replace("metadata.csv", ""), "wavs", cols[0] + ".wav")
                if os.path.isfile(wav_file):
                    text = cols[1].strip()
                    items.append(
                        {"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path}
                    )
                else:
                    # M-AI-Labs have some missing samples, so just print the warning
                    logger.warning("File %s does not exist!", wav_file)
    return items


def ljspeech(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize the LJSpeech meta data file to TTS format.

    Website: https://keithito.com/LJ-Speech-Dataset/

    Expected metadata format (pipe-delimited)::

        LJ001-0001|This is the transcription.|This is the normalized transcription.
        LJ001-0002|It'll be $16 sir.|It'll be sixteen dollars sir.

    Note: ``.wav`` is automatically appended to the utterance ID and only the text
    of the third column is used.

    Audio files should be in: ``{root_path}/wavs/{filename}.wav``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            if len(cols) < 3:
                msg = "LJSpeech format expects 3 pipe-delimited columns in the metadata file."
                raise IndexError(msg)
            text = cols[2]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def ljspeech_test(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file for TTS testing
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, encoding="utf-8") as ttf:
        speaker_id = 0
        for idx, line in enumerate(ttf):
            # 2 samples per speaker to avoid eval split issues
            if idx % 2 == 0:
                speaker_id += 1
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2]
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": f"ljspeech-{speaker_id}", "root_path": root_path}
            )
    return items


def thorsten(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize the Thorsten German TTS meta data file to TTS format.

    Website: https://github.com/thorstenMueller/Thorsten-Voice

    Expected metadata format (pipe-delimited)::

        file001|This is the German transcript.
        file002|Another sentence here.

    Note: ``.wav`` is automatically appended to the file ID.

    Audio files should be in: ``{root_path}/wavs/{filename}.wav``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "thorsten"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def sam_accenture(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize the Sam Accenture Non-Binary Voice meta data file to TTS format.

    Website: https://github.com/Sam-Accenture-Non-Binary-Voice/non-binary-voice-files

    Expected metadata format (XML file, typically ``voice_over_recordings/recording_script.xml``)::

        <recording>
            <fileid id="file001">This is the transcript.</fileid>
            <fileid id="file002">Another sentence here.</fileid>
        </recording>

    Note: ``.wav`` is automatically appended to the file ID.

    Audio files should be in: ``{root_path}/vo_voice_quality_transformation/{id}.wav``
    """
    xml_file = os.path.join(root_path, "voice_over_recordings", meta_file)
    xml_root = ET.parse(xml_file).getroot()
    items = []
    speaker_name = "sam_accenture"
    for item in xml_root.findall("./fileid"):
        text = item.text
        wav_file = os.path.join(root_path, "vo_voice_quality_transformation", item.get("id") + ".wav")
        if not os.path.exists(wav_file):
            logger.warning("%s in metafile does not exist. Skipping...", wav_file)
            continue
        items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def ruslan(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize the RUSLAN meta data file to TTS format.

    Website: https://ruslan-corpus.github.io/

    Expected metadata format (pipe-delimited)::

        file001|This is the Russian transcript.
        file002|Another sentence here.

    Note: ``.wav`` is automatically appended to the file ID.

    Audio files should be in: ``{root_path}/RUSLAN/{filename}.wav``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ruslan"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "RUSLAN", cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def css10(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize the CSS10 dataset file to TTS format.

    Website: https://github.com/Kyubyong/css10

    Expected metadata format (pipe-delimited)::

        audio/file001.wav|This is the transcript.
        audio/file002.wav|Another sentence here.

    Note: The full audio file path (relative to ``root_path``) is included in the metadata.

    Audio files are located at: ``{root_path}/{filepath}``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "css10"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def nancy(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize the Nancy meta data file to TTS format.

    Website: https://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/

    Expected metadata format (space-delimited with quoted text)::

        ( file001 "This is the transcript." )
        ( file002 "Another sentence here." )

    Note: ``.wav`` is automatically appended to the file ID.

    Audio files should be in: ``{root_path}/wavn/{filename}.wav``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "nancy"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            utt_id = line.split()[1]
            text = line[line.find('"') + 1 : line.rfind('"') - 1]
            wav_file = os.path.join(root_path, "wavn", utt_id + ".wav")
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def common_voice(root_path, meta_file, ignored_speakers=None):
    """Normalize the Mozilla Common Voice meta data file to TTS format.

    Website: https://commonvoice.mozilla.org/en/datasets

    Expected metadata format (tab-delimited with header)::

        client_id	path	sentence	...
        speaker001	file001.mp3	This is the transcript.	...
        speaker002	file002.mp3	Another sentence here.	...

    Audio files should be in: ``{root_path}/clips/{filename}.mp3``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            if line.startswith("client_id"):
                continue
            cols = line.split("\t")
            text = cols[2]
            speaker_name = cols[0]
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker_name in ignored_speakers:
                    continue
            wav_file = os.path.join(root_path, "clips", cols[1])
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": "MCV_" + speaker_name, "root_path": root_path}
            )
    return items


def libri_tts(root_path, meta_files=None, ignored_speakers=None):
    """Normalize the LibriTTS meta data file to TTS format.

    Website: https://www.openslr.org/60/

    Expected metadata format (tab-delimited, no header)::

        84_121550_000007_000000	It's $50 sir.	It's fifty dollars sir.
        84_121550_000007_000000	Call me at 5pm.	Call me at five P M.

    Note: This dataset automatically searches for all ``*trans.tsv`` files recursively
    in ``root_path`` unless ``meta_files`` is specified. ``.wav`` is automatically appended.
    Only the normalized transcript (third column) is used. The speaker name is
    the first part of the utterance ID.

    Audio files should be in: ``{root_path}/{speaker}/{chapter}/{filename}.wav``
    """
    items = []
    if not meta_files:
        meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    else:
        if isinstance(meta_files, str):
            meta_files = [os.path.join(root_path, meta_files)]

    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split(".")[0]
        with open(meta_file, encoding="utf-8") as ttf:
            for line in ttf:
                cols = line.split("\t")
                file_name = cols[0]
                speaker_name, chapter_id, *_ = cols[0].split("_")
                _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
                wav_file = os.path.join(_root_path, file_name + ".wav")
                text = cols[2]
                # ignore speakers
                if isinstance(ignored_speakers, list):
                    if speaker_name in ignored_speakers:
                        continue
                items.append(
                    {
                        "text": text,
                        "audio_file": wav_file,
                        "speaker_name": f"LTTS_{speaker_name}",
                        "root_path": root_path,
                    }
                )
    for item in items:
        assert os.path.exists(item["audio_file"]), f" [!] wav files don't exist - {item['audio_file']}"
    return items


def custom_turkish(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize a custom Turkish dataset to TTS format.

    Expected metadata format (pipe-delimited)::

        file001|Bu bir transkript.
        file002|Başka bir cümle.

    Note: ``.wav`` is automatically appended to the file ID.

    Audio files should be in: ``{root_path}/wavs/{filename}.wav``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "turkish-female"
    skipped_files = []
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0].strip() + ".wav")
            if not os.path.exists(wav_file):
                skipped_files.append(wav_file)
                continue
            text = cols[1].strip()
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    logger.warning("%d files skipped. They don't exist...")
    return items


def brspeech(root_path, meta_file, ignored_speakers=None):
    """Normalize the BRSpeech 3.0 beta dataset to TTS format.

    Website: https://github.com/freds0/BRSpeech-Dataset

    Expected metadata format (pipe-delimited with header)::

        wav_filename|transcript_raw|transcript|speaker_id
        audio/speaker1/file001.wav|Raw text|Normalized text|speaker1
        audio/speaker2/file002.wav|Raw text|Normalized text|speaker2

    Note: Only the normalized transcript (third column) is used.

    Audio files are located at: ``{root_path}/{wav_filename}``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            if line.startswith("wav_filename"):
                continue
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[2]
            speaker_id = cols[3]
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker_id in ignored_speakers:
                    continue
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_id, "root_path": root_path})
    return items


def vctk(root_path, meta_files=None, wavs_path="wav48_silence_trimmed", mic="mic1", ignored_speakers=None):
    """Normalize the VCTK dataset v0.92 to TTS format.

    Website: https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip

    Expected metadata format (individual text files per utterance)::

        txt/p225/p225_001.txt: "Please call Stella."
        txt/p225/p225_002.txt: "Ask her to bring these things with her from the store."

    Note: This dataset automatically searches for all ``.txt`` files in ``{root_path}/txt/``.
    Each text file contains a single line of transcription. Audio files use ``.flac`` format.

    This dataset has 2 recordings per speaker (``mic1`` and ``mic2``):

    - **mic1**: Omni-directional microphone (DPA 4035). Same as previous VCTK versions.
    - **mic2**: Small diaphragm condenser microphone (Sennheiser MKH 800). Speakers p280 and p315 have technical issues.

    Audio files should be in: ``{root_path}/{wavs_path}/{speaker_id}/{file_id}_{mic}.flac``
    """
    file_ext = "flac"
    items = []
    meta_files = glob(f"{os.path.join(root_path, 'txt')}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, encoding="utf-8") as file_text:
            text = file_text.readlines()[0]
        # p280 has no mic2 recordings
        if speaker_id == "p280":
            wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + f"_mic1.{file_ext}")
        else:
            wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + f"_{mic}.{file_ext}")
        if os.path.exists(wav_file):
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": "VCTK_" + speaker_id, "root_path": root_path}
            )
        else:
            logger.warning("Wav file doesn't exist - %s", wav_file)
    return items


def vctk_old(root_path, meta_files=None, wavs_path="wav48", ignored_speakers=None):
    """Normalize the older VCTK dataset to TTS format.

    Website: https://datashare.ed.ac.uk/handle/10283/2651

    Expected metadata format (individual text files per utterance)::

        txt/p225/p225_001.txt: "Please call Stella."
        txt/p225/p225_002.txt: "Ask her to bring these things with her from the store."

    Note: This dataset automatically searches for all ``.txt`` files in ``{root_path}/txt/``.
    Each text file contains a single line of transcription. Audio files use ``.wav`` format.

    Audio files should be in: ``{root_path}/{wavs_path}/{speaker_id}/{file_id}.wav``
    """
    items = []
    meta_files = glob(f"{os.path.join(root_path, 'txt')}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, encoding="utf-8") as file_text:
            text = file_text.readlines()[0]
        wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + ".wav")
        items.append(
            {"text": text, "audio_file": wav_file, "speaker_name": "VCTK_old_" + speaker_id, "root_path": root_path}
        )
    return items


def synpaflex(root_path, metafiles=None, **kwargs):  # pylint: disable=unused-argument
    """Normalize the SynPaFlex dataset to TTS format.

    Website: http://synpaflex.irisa.fr/corpus/

    Expected metadata format (individual text files per utterance)::

        wav/file001.wav with corresponding txt/file001.txt: "This is the transcript."
        wav/file002.wav with corresponding txt/file002.txt: "Another sentence here."

    Note: This dataset automatically searches for all ``.wav`` files recursively in ``root_path``.
    For each wav file, it looks for a corresponding ``.txt`` file in a ``txt`` folder.

    Audio files are located at: ``{root_path}/**/*.wav``
    """
    items = []
    speaker_name = "synpaflex"
    root_path = os.path.join(root_path, "")
    wav_files = glob(f"{root_path}**/*.wav", recursive=True)
    for wav_file in wav_files:
        if os.sep + "wav" + os.sep in wav_file:
            txt_file = wav_file.replace("wav", "txt")
        else:
            txt_file = os.path.join(
                os.path.dirname(wav_file), "txt", os.path.basename(wav_file).replace(".wav", ".txt")
            )
        if os.path.exists(txt_file) and os.path.exists(wav_file):
            with open(txt_file, encoding="utf-8") as file_text:
                text = file_text.readlines()[0]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def open_bible(root_path, meta_files="train", ignore_digits_sentences=True, ignored_speakers=None):
    """Normalize the BibleTTS dataset to TTS format.

    Website: https://masakhane-io.github.io/bibleTTS/

    Expected metadata format (individual text files per utterance)::

        train/speaker1/file001.txt: "This is the transcript."
        train/speaker1/file002.txt: "Another sentence here."

    Note: This dataset automatically searches for all ``.txt`` files in ``{root_path}/{split}/``.
    Each text file contains a single line of transcription. Audio files use ``.flac`` format.
    By default, sentences containing digits are ignored.

    Audio files should be in: ``{root_path}/{split}/{speaker_id}/{file_id}.flac``
    """
    items = []
    split_dir = meta_files
    meta_files = glob(f"{os.path.join(root_path, split_dir)}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, encoding="utf-8") as file_text:
            text = file_text.readline().replace("\n", "")
        # ignore sentences that contains digits
        if ignore_digits_sentences and any(map(str.isdigit, text)):
            continue
        wav_file = os.path.join(root_path, split_dir, speaker_id, file_id + ".flac")
        items.append({"text": text, "audio_file": wav_file, "speaker_name": "OB_" + speaker_id, "root_path": root_path})
    return items


def mls(root_path, meta_files=None, ignored_speakers=None):
    """Normalize the Multilingual LibriSpeech (MLS) dataset to TTS format.

    Website: http://www.openslr.org/94/

    Expected metadata format (tab-delimited, no header)::

        speaker_book_utterance	This is the transcript.
        1001_1234_000001	Another sentence here.

    Note: ``.wav`` is automatically appended to the file ID. Speaker and book IDs
    are extracted from the filename.

    Audio files should be in: ``{root_path}/{meta_files_dir}/audio/{speaker}/{book}/{filename}.wav``
    """
    items = []
    with open(os.path.join(root_path, meta_files), encoding="utf-8") as meta:
        for line in meta:
            file, text = line.split("\t")
            text = text[:-1]
            speaker, book, *_ = file.split("_")
            wav_file = os.path.join(root_path, os.path.dirname(meta_files), "audio", speaker, book, file + ".wav")
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker in ignored_speakers:
                    continue
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": "MLS_" + speaker, "root_path": root_path}
            )
    return items


# ======================================== VOX CELEB ===========================================
def voxceleb2(root_path, meta_file=None, **kwargs):  # pylint: disable=unused-argument
    """Normalize the VoxCeleb2 dataset to TTS format.

    Website: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html

    Note: This dataset automatically searches for all ``.wav`` files recursively in ``root_path``
    and creates a cached metadata file. No transcriptions are provided (used for speaker encoder training).
    Speaker IDs are extracted from the folder structure.

    Audio files should be in: ``{root_path}/id{speaker_id}/{video_id}/{utterance_id}.wav``
    """
    return _voxcel_x(root_path, meta_file, voxcel_idx="2")


def voxceleb1(root_path, meta_file=None, **kwargs):  # pylint: disable=unused-argument
    """Normalize the VoxCeleb1 dataset to TTS format.

    Website: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html

    Note: This dataset automatically searches for all ``.wav`` files recursively in ``root_path``
    and creates a cached metadata file. No transcriptions are provided (used for speaker encoder training).
    Speaker IDs are extracted from the folder structure.

    Audio files should be in: ``{root_path}/id{speaker_id}/{video_id}/{utterance_id}.wav``
    """
    return _voxcel_x(root_path, meta_file, voxcel_idx="1")


def _voxcel_x(root_path, meta_file, voxcel_idx):
    assert voxcel_idx in ["1", "2"]
    expected_count = 148_000 if voxcel_idx == "1" else 1_000_000
    voxceleb_path = Path(root_path)
    cache_to = voxceleb_path / f"metafile_voxceleb{voxcel_idx}.csv"
    cache_to.parent.mkdir(exist_ok=True)

    # if not exists meta file, crawl recursively for 'wav' files
    if meta_file is not None:
        with open(str(meta_file), encoding="utf-8") as f:
            return [x.strip().split("|") for x in f.readlines()]

    elif not cache_to.exists():
        cnt = 0
        meta_data = []
        wav_files = voxceleb_path.rglob("**/*.wav")
        for path in tqdm(
            wav_files,
            desc=f"Building VoxCeleb {voxcel_idx} Meta file ... this needs to be done only once.",
            total=expected_count,
        ):
            speaker_id = str(Path(path).parent.parent.stem)
            assert speaker_id.startswith("id")
            text = None  # VoxCel does not provide transciptions, and they are not needed for training the SE
            meta_data.append(f"{text}|{path}|voxcel{voxcel_idx}_{speaker_id}\n")
            cnt += 1
        with open(str(cache_to), "w", encoding="utf-8") as f:
            f.write("".join(meta_data))
        if cnt < expected_count:
            raise ValueError(f"Found too few instances for Voxceleb. Should be around {expected_count}, is: {cnt}")

    with open(str(cache_to), encoding="utf-8") as f:
        return [x.strip().split("|") for x in f.readlines()]


def emotion(root_path, meta_file, ignored_speakers=None):
    """Normalize a generic emotion dataset to TTS format.

    Expected metadata format (comma-delimited with header)::

        file_path,speaker_id,emotion_id
        audio/speaker1/file001.wav,speaker1,happy
        audio/speaker2/file002.wav,speaker2,sad

    Note: No text transcriptions are included, only emotion labels.

    Audio files are located at: ``{root_path}/{file_path}``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            if line.startswith("file_path"):
                continue
            cols = line.split(",")
            wav_file = os.path.join(root_path, cols[0])
            speaker_id = cols[1]
            emotion_id = cols[2].replace("\n", "")
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker_id in ignored_speakers:
                    continue
            items.append(
                {"audio_file": wav_file, "speaker_name": speaker_id, "emotion_name": emotion_id, "root_path": root_path}
            )
    return items


def baker(root_path: str, meta_file: str, **kwargs) -> list[list[str]]:  # pylint: disable=unused-argument
    """Normalize the Baker Chinese TTS dataset to TTS format.

    Website: https://www.data-baker.com/data/index/TNtts/

    Expected metadata format (pipe-delimited)::

        000001|This is the Chinese transcript.
        000002|Another sentence here.

    Audio files should be in: ``{root_path}/clips_22/{filename}.wav``

    Args:
        root_path (str): path to the baker dataset
        meta_file (str): name of the meta dataset containing names of wav to select and the transcript of the sentence
    Returns:
        List[List[str]]: List of (text, wav_path, speaker_name) associated with each sentences
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "baker"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            wav_name, text = line.rstrip("\n").split("|")
            wav_path = os.path.join(root_path, "clips_22", wav_name)
            items.append({"text": text, "audio_file": wav_path, "speaker_name": speaker_name, "root_path": root_path})
    return items


def kokoro(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize the Kokoro Japanese Speech dataset to TTS format.

    Website: https://github.com/kaiidams/Kokoro-Speech-Dataset

    Expected metadata format (pipe-delimited)::

        file001|raw transcript|normalized transcript
        file002|raw transcript|normalized transcript

    Note: ``.wav`` is automatically appended to the file ID. The normalized transcript
    (third column) is used with spaces removed.

    Audio files should be in: ``{root_path}/wavs/{filename}.wav``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "kokoro"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2].replace(" ", "")
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def kss(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize the Korean Single Speaker (KSS) dataset to TTS format.

    Website: https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset

    Expected metadata format (pipe-delimited)::

        audio/file001.wav|raw transcript|normalized transcript
        audio/file002.wav|raw transcript|normalized transcript

    Note: The full audio file path (relative to ``root_path``) is included in the metadata.
    Only the normalized transcript (third column) is used.

    Audio files are located at: ``{root_path}/{filepath}``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "kss"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[2]  # cols[1] => 6월, cols[2] => 유월
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def bel_tts_formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize the Belarusian TTS dataset to TTS format.

    Expected metadata format (pipe-delimited)::

        audio/file001.wav|This is the Belarusian transcript.
        audio/file002.wav|Another sentence here.

    Note: The full audio file path (relative to ``root_path``) is included in the metadata.

    Audio files are located at: ``{root_path}/{filepath}``
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "bel_tts"
    with open(txt_file, encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


### Registrations
register_formatter("cml_tts", cml_tts)
register_formatter("coqui", coqui)
register_formatter("tweb", tweb)
register_formatter("mozilla", mozilla)
register_formatter("mozilla_de", mozilla_de)
register_formatter("mailabs", mailabs)
register_formatter("ljspeech", ljspeech)
register_formatter("ljspeech_test", ljspeech_test)
register_formatter("thorsten", thorsten)
register_formatter("sam_accenture", sam_accenture)
register_formatter("ruslan", ruslan)
register_formatter("css10", css10)
register_formatter("nancy", nancy)
register_formatter("common_voice", common_voice)
register_formatter("libri_tts", libri_tts)
register_formatter("custom_turkish", custom_turkish)
register_formatter("brspeech", brspeech)
register_formatter("vctk", vctk)
register_formatter("vctk_old", vctk_old)
register_formatter("synpaflex", synpaflex)
register_formatter("open_bible", open_bible)
register_formatter("mls", mls)
register_formatter("voxceleb2", voxceleb2)
register_formatter("voxceleb1", voxceleb1)
register_formatter("emotion", emotion)
register_formatter("baker", baker)
register_formatter("kokoro", kokoro)
register_formatter("kss", kss)
register_formatter("bel_tts_formatter", bel_tts_formatter)
