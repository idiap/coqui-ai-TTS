# Copyright (C) 2020 ATHENA AUTHORS; Yiping Peng; Ne Luo
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
"""voxceleb 1 & 2"""

import csv
import hashlib
import logging
import os
import subprocess
import sys
import zipfile

import soundfile as sf

from TTS.utils.generic_utils import ConsoleFormatter, setup_logger

logger = logging.getLogger(__name__)

SUBSETS = {
    "vox1_dev_wav": [
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad",
    ],
    "vox1_test_wav": ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip"],
    "vox2_dev_aac": [
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partab",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partac",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partad",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partae",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaf",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partag",
        "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partah",
    ],
    "vox2_test_aac": ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip"],
}

MD5SUM = {
    "vox1_dev_wav": "ae63e55b951748cc486645f532ba230b",
    "vox2_dev_aac": "bbc063c46078a602ca71605645c2a402",
    "vox1_test_wav": "185fdc63c3c739954633d50379a3d102",
    "vox2_test_aac": "0d2b3ea430a821c33263b5ea37ede312",
}

USER = {"user": "", "password": ""}

speaker_id_dict = {}


def download_and_extract(directory, subset, urls):
    """Download and extract the given split of dataset.

    Args:
        directory: the directory where to put the downloaded data.
        subset: subset name of the corpus.
        urls: the list of urls to download the data file.
    """
    os.makedirs(directory, exist_ok=True)

    try:
        for url in urls:
            zip_filepath = os.path.join(directory, url.split("/")[-1])
            if os.path.exists(zip_filepath):
                continue
            logger.info("Downloading %s to %s", url, zip_filepath)
            subprocess.call(
                "wget {} --user {} --password {} -O {}".format(url, USER["user"], USER["password"], zip_filepath),
                shell=True,
            )

            statinfo = os.stat(zip_filepath)
            logger.info("Successfully downloaded %s, size(bytes): %d", url, statinfo.st_size)

        # concatenate all parts into zip files
        if ".zip" not in zip_filepath:
            zip_filepath = "_".join(zip_filepath.split("_")[:-1])
            subprocess.call(f"cat {zip_filepath}* > {zip_filepath}.zip", shell=True)
            zip_filepath += ".zip"
        extract_path = zip_filepath.strip(".zip")

        # check zip file md5sum
        with open(zip_filepath, "rb") as f_zip:
            md5 = hashlib.md5(f_zip.read()).hexdigest()
        if md5 != MD5SUM[subset]:
            raise ValueError(f"md5sum of {zip_filepath} mismatch")

        with zipfile.ZipFile(zip_filepath, "r") as zfile:
            zfile.extractall(directory)
            extract_path_ori = os.path.join(directory, zfile.infolist()[0].filename)
            subprocess.call(f"mv {extract_path_ori} {extract_path}", shell=True)
    finally:
        # os.remove(zip_filepath)
        pass


def exec_cmd(cmd):
    """Run a command in a subprocess.
    Args:
        cmd: command line to be executed.
    Return:
        int, the return code.
    """
    try:
        retcode = subprocess.call(cmd, shell=True)
        if retcode < 0:
            logger.info("Child was terminated by signal %d", retcode)
    except OSError as e:
        logger.info("Execution failed: %s", e)
        retcode = -999
    return retcode


def decode_aac_with_ffmpeg(aac_file, wav_file):
    """Decode a given AAC file into WAV using ffmpeg.
    Args:
        aac_file: file path to input AAC file.
        wav_file: file path to output WAV file.
    Return:
        bool, True if success.
    """
    cmd = f"ffmpeg -i {aac_file} {wav_file}"
    logger.info("Decoding aac file using command line: %s", cmd)
    ret = exec_cmd(cmd)
    if ret != 0:
        logger.error("Failed to decode aac file with retcode %s", ret)
        logger.error("Please check your ffmpeg installation.")
        return False
    return True


def convert_audio_and_make_label(input_dir, subset, output_dir, output_file):
    """Optionally convert AAC to WAV and make speaker labels.
    Args:
        input_dir: the directory which holds the input dataset.
        subset: the name of the specified subset. e.g. vox1_dev_wav
        output_dir: the directory to place the newly generated csv files.
        output_file: the name of the newly generated csv file. e.g. vox1_dev_wav.csv
    """

    logger.info("Preprocessing audio and label for subset %s", subset)
    source_dir = os.path.join(input_dir, subset)

    files = []
    # Convert all AAC file into WAV format. At the same time, generate the csv
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext.lower() == ".wav":
                _, ext2 = os.path.splitext(name)
                if ext2:
                    continue
                wav_file = os.path.join(root, filename)
            elif ext.lower() == ".m4a":
                # Convert AAC to WAV.
                aac_file = os.path.join(root, filename)
                wav_file = aac_file + ".wav"
                if not os.path.exists(wav_file):
                    if not decode_aac_with_ffmpeg(aac_file, wav_file):
                        raise RuntimeError("Audio decoding failed.")
            else:
                continue
            speaker_name = root.split(os.path.sep)[-2]
            if speaker_name not in speaker_id_dict:
                num = len(speaker_id_dict)
                speaker_id_dict[speaker_name] = num
            # wav_filesize = os.path.getsize(wav_file)
            wav_length = len(sf.read(wav_file)[0])
            files.append((os.path.abspath(wav_file), wav_length, speaker_id_dict[speaker_name], speaker_name))

    # Write to CSV file which contains four columns:
    # "wav_filename", "wav_length_ms", "speaker_id", "speaker_name".
    csv_file_path = os.path.join(output_dir, output_file)
    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["wav_filename", "wav_length_ms", "speaker_id", "speaker_name"])
        for wav_file in files:
            writer.writerow(wav_file)
    logger.info("Successfully generated csv file %s", csv_file_path)


def processor(directory, subset, force_process):
    """download and process"""
    urls = SUBSETS
    if subset not in urls:
        raise ValueError(subset, "is not in voxceleb")

    subset_csv = os.path.join(directory, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        return subset_csv

    logger.info("Downloading and process the voxceleb in %s", directory)
    logger.info("Preparing subset %s", subset)
    download_and_extract(directory, subset, urls[subset])
    convert_audio_and_make_label(directory, subset, directory, subset + ".csv")
    logger.info("Finished downloading and processing")
    return subset_csv


if __name__ == "__main__":
    setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())
    if len(sys.argv) != 4:
        print("Usage: python prepare_data.py save_directory user password")
        sys.exit()

    DIR, USER["user"], USER["password"] = sys.argv[1], sys.argv[2], sys.argv[3]
    for SUBSET in SUBSETS:
        processor(DIR, SUBSET, False)
