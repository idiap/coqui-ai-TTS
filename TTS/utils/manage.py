import json
import logging
import os
import re
import tarfile
import zipfile
from contextlib import nullcontext
from pathlib import Path
from shutil import copyfile, rmtree
from typing import Any, TypedDict

import fsspec
import requests
from tqdm import tqdm
from trainer.io import get_user_data_dir
from typing_extensions import Required

from TTS.config import load_config, read_json_with_comments
from TTS.vc.configs.knnvc_config import KNNVCConfig

logger = logging.getLogger(__name__)


class ModelItem(TypedDict, total=False):
    model_name: Required[str]
    model_type: Required[str]
    description: str
    license: str
    author: str
    contact: str
    commit: str | None
    model_hash: str
    tos_required: bool
    default_vocoder: str | None
    model_url: str | list[str]
    github_rls_url: str | list[str]
    hf_url: list[str]


LICENSE_URLS = {
    "cc by-nc-nd 4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "mpl": "https://www.mozilla.org/en-US/MPL/2.0/",
    "mpl2": "https://www.mozilla.org/en-US/MPL/2.0/",
    "mpl 2.0": "https://www.mozilla.org/en-US/MPL/2.0/",
    "mit": "https://choosealicense.com/licenses/mit/",
    "apache 2.0": "https://choosealicense.com/licenses/apache-2.0/",
    "apache2": "https://choosealicense.com/licenses/apache-2.0/",
    "cc-by-sa 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cpml": "https://coqui.ai/cpml.txt",
}


class ModelManager:
    """Manage TTS models defined in .models.json.
    It provides an interface to list and download
    models defines in '.model.json'

    Models are downloaded under '.TTS' folder in the user's
    home path.

    Args:
        models_file (str or Path): path to .model.json file. Defaults to None.
        output_prefix (str or Path): prefix to `tts` to download models. Defaults to None
        progress_bar (bool): print a progress bar when donwloading a file. Defaults to False.
    """

    def __init__(
        self,
        models_file: str | os.PathLike[Any] | None = None,
        output_prefix: str | os.PathLike[Any] | None = None,
        progress_bar: bool = False,
    ) -> None:
        super().__init__()
        self.progress_bar = progress_bar
        if output_prefix is None:
            self.output_prefix = get_user_data_dir("tts")
        else:
            self.output_prefix = Path(output_prefix) / "tts"
        self.models_dict = {}
        if models_file is not None:
            self.read_models_file(models_file)
        else:
            # try the default location
            path = Path(__file__).parent / "../.models.json"
            self.read_models_file(path)

    def read_models_file(self, file_path: str | os.PathLike[Any]) -> None:
        """Read .models.json as a dict

        Args:
            file_path (str): path to .models.json.
        """
        self.models_dict = read_json_with_comments(file_path)

    def _list_models(self, model_type: str, model_count: int = 0) -> list[str]:
        logger.info("")
        logger.info("Name format: type/language/dataset/model")
        model_list = []
        for lang in self.models_dict[model_type]:
            for dataset in self.models_dict[model_type][lang]:
                for model in self.models_dict[model_type][lang][dataset]:
                    model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
                    output_path = Path(self.output_prefix) / model_full_name
                    downloaded = " [already downloaded]" if output_path.is_dir() else ""
                    logger.info(" %2d: %s/%s/%s/%s%s", model_count, model_type, lang, dataset, model, downloaded)
                    model_list.append(f"{model_type}/{lang}/{dataset}/{model}")
                    model_count += 1
        return model_list

    def _list_for_model_type(self, model_type: str) -> list[str]:
        models_name_list = []
        model_count = 1
        models_name_list.extend(self._list_models(model_type, model_count))
        return models_name_list

    def list_models(self) -> list[str]:
        models_name_list = []
        model_count = 1
        for model_type in self.models_dict:
            model_list = self._list_models(model_type, model_count)
            models_name_list.extend(model_list)
        logger.info("")
        logger.info("Path to downloaded models: %s", self.output_prefix)
        return models_name_list

    def log_model_details(self, model_type: str, lang: str, dataset: str, model: str) -> None:
        logger.info("Model type: %s", model_type)
        logger.info("Language supported: %s", lang)
        logger.info("Dataset used: %s", dataset)
        logger.info("Model name: %s", model)
        if "description" in self.models_dict[model_type][lang][dataset][model]:
            logger.info("Description: %s", self.models_dict[model_type][lang][dataset][model]["description"])
        else:
            logger.info("Description: coming soon")
        if "default_vocoder" in self.models_dict[model_type][lang][dataset][model]:
            logger.info(
                "Default vocoder: %s",
                self.models_dict[model_type][lang][dataset][model]["default_vocoder"],
            )

    def model_info_by_idx(self, model_query: str) -> None:
        """Print the description of the model from .models.json file using model_query_idx

        Args:
            model_query (str): <model_tye>/<model_query_idx>
        """
        model_name_list = []
        model_type, model_query_idx = model_query.split("/")
        try:
            model_query_idx = int(model_query_idx)
            if model_query_idx <= 0:
                logger.error("model_query_idx [%d] should be a positive integer!", model_query_idx)
                return
        except (TypeError, ValueError):
            logger.error("model_query_idx [%s] should be an integer!", model_query_idx)
            return
        model_count = 0
        if model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                    for model in self.models_dict[model_type][lang][dataset]:
                        model_name_list.append(f"{model_type}/{lang}/{dataset}/{model}")
                        model_count += 1
        else:
            logger.error("Model type %s does not exist in the list.", model_type)
            return
        if model_query_idx > model_count:
            logger.error("model_query_idx exceeds the number of available models [%d]", model_count)
        else:
            model_type, lang, dataset, model = model_name_list[model_query_idx - 1].split("/")
            self.log_model_details(model_type, lang, dataset, model)

    def model_info_by_full_name(self, model_query_name: str) -> None:
        """Print the description of the model from .models.json file using model_full_name

        Args:
            model_query_name (str): Format is <model_type>/<language>/<dataset>/<model_name>
        """
        model_type, lang, dataset, model = model_query_name.split("/")
        if model_type not in self.models_dict:
            logger.error("Model type %s does not exist in the list.", model_type)
            return
        if lang not in self.models_dict[model_type]:
            logger.error("Language %s does not exist for %s.", lang, model_type)
            return
        if dataset not in self.models_dict[model_type][lang]:
            logger.error("Dataset %s does not exist for %s/%s.", dataset, model_type, lang)
            return
        if model not in self.models_dict[model_type][lang][dataset]:
            logger.error("Model %s does not exist for %s/%s/%s.", model, model_type, lang, dataset)
            return
        self.log_model_details(model_type, lang, dataset, model)

    def list_tts_models(self) -> list[str]:
        """Print all `TTS` models and return a list of model names

        Format is `language/dataset/model`
        """
        return self._list_for_model_type("tts_models")

    def list_vocoder_models(self) -> list[str]:
        """Print all the `vocoder` models and return a list of model names

        Format is `language/dataset/model`
        """
        return self._list_for_model_type("vocoder_models")

    def list_vc_models(self) -> list[str]:
        """Print all the voice conversion models and return a list of model names

        Format is `language/dataset/model`
        """
        return self._list_for_model_type("voice_conversion_models")

    def list_langs(self) -> None:
        """Print all the available languages"""
        logger.info("Name format: type/language")
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                logger.info("  %s/%s", model_type, lang)

    def list_datasets(self) -> None:
        """Print all the datasets"""
        logger.info("Name format: type/language/dataset")
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                    logger.info("  %s/%s/%s", model_type, lang, dataset)

    @staticmethod
    def print_model_license(model_item: ModelItem) -> None:
        """Print the license of a model

        Args:
            model_item (dict): model item in the models.json
        """
        if "license" in model_item and model_item["license"].strip() != "":
            logger.info("Model's license - %s", model_item["license"])
            if model_item["license"].lower() in LICENSE_URLS:
                logger.info("Check %s for more info.", LICENSE_URLS[model_item["license"].lower()])
            else:
                logger.info("Check https://opensource.org/licenses for more info.")
        else:
            logger.info("Model's license - No license information available")

    def _download_github_model(self, model_item: ModelItem, output_path: Path) -> None:
        if isinstance(model_item["github_rls_url"], list):
            self._download_model_files(model_item["github_rls_url"], output_path, self.progress_bar)
        else:
            self._download_zip_file(model_item["github_rls_url"], output_path, self.progress_bar)

    def _download_hf_model(self, model_item: ModelItem, output_path: Path) -> None:
        if isinstance(model_item["hf_url"], list):
            self._download_model_files(model_item["hf_url"], output_path, self.progress_bar)
        else:
            self._download_zip_file(model_item["hf_url"], output_path, self.progress_bar)

    def download_fairseq_model(self, model_name: str, output_path: Path) -> None:
        URI_PREFIX = "https://dl.fbaipublicfiles.com/mms/tts/"
        _, lang, _, _ = model_name.split("/")
        model_download_uri = os.path.join(URI_PREFIX, f"{lang}.tar.gz")
        self._download_tar_file(model_download_uri, output_path, self.progress_bar)

    @staticmethod
    def set_model_url(model_item: ModelItem) -> ModelItem:
        model_item["model_url"] = ""
        if "github_rls_url" in model_item:
            model_item["model_url"] = model_item["github_rls_url"]
        elif "hf_url" in model_item:
            model_item["model_url"] = model_item["hf_url"]
        elif "fairseq" in model_item.get("model_name", ""):
            model_item["model_url"] = "https://dl.fbaipublicfiles.com/mms/tts/"
        elif "xtts" in model_item.get("model_name", ""):
            model_item["model_url"] = "https://huggingface.co/coqui/"
        return model_item

    def _set_model_item(self, model_name: str) -> tuple[ModelItem, str, str, str | None]:
        # fetch model info from the dict
        if "fairseq" in model_name:
            model_type, lang, dataset, model = model_name.split("/")
            model_item: ModelItem = {
                "model_name": model_name,
                "model_type": "tts_models",
                "license": "CC BY-NC 4.0",
                "default_vocoder": None,
                "author": "fairseq",
                "description": "this model is released by Meta under Fairseq repo. Visit https://github.com/facebookresearch/fairseq/tree/main/examples/mms for more info.",
            }
        elif "xtts" in model_name and len(model_name.split("/")) != 4:
            # loading xtts models with only model name (e.g. xtts_v2.0.2)
            # check model name has the version number with regex
            version_regex = r"v\d+\.\d+\.\d+"
            if re.search(version_regex, model_name):
                model_version = model_name.split("_")[-1]
            else:
                model_version = "main"
            model_type = "tts_models"
            lang = "multilingual"
            dataset = "multi-dataset"
            model = model_name
            model_item = {
                "model_name": model_name,
                "model_type": model_type,
                "default_vocoder": None,
                "license": "CPML",
                "contact": "info@coqui.ai",
                "tos_required": True,
                "hf_url": [
                    f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_version}/model.pth",
                    f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_version}/config.json",
                    f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_version}/vocab.json",
                    f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_version}/hash.md5",
                    f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_version}/speakers_xtts.pth",
                ],
            }
        else:
            # get model from models.json
            model_type, lang, dataset, model = model_name.split("/")
            model_item = self.models_dict[model_type][lang][dataset][model]
            model_item["model_type"] = model_type

        model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
        md5hash = model_item["model_hash"] if "model_hash" in model_item else None
        model_item = self.set_model_url(model_item)
        return model_item, model_full_name, model, md5hash

    @staticmethod
    def ask_tos(model_full_path: Path) -> bool:
        """Ask the user to agree to the terms of service"""
        tos_path = model_full_path / "tos_agreed.txt"
        print(" > You must confirm the following:")
        print(' | > "I have purchased a commercial license from Coqui: licensing@coqui.ai"')
        print(' | > "Otherwise, I agree to the terms of the non-commercial CPML: https://coqui.ai/cpml" - [y/n]')
        answer = input(" | | > ")
        if answer.lower() == "y":
            with open(tos_path, "w", encoding="utf-8") as f:
                f.write("I have read, understood and agreed to the Terms and Conditions.")
            return True
        return False

    @staticmethod
    def tos_agreed(model_item: ModelItem, model_full_path: Path) -> bool:
        """Check if the user has agreed to the terms of service"""
        if "tos_required" in model_item and model_item["tos_required"]:
            tos_path = os.path.join(model_full_path, "tos_agreed.txt")
            if os.path.exists(tos_path) or os.environ.get("COQUI_TOS_AGREED") == "1":
                return True
            return False
        return True

    def create_dir_and_download_model(self, model_name: str, model_item: ModelItem, output_path: Path) -> None:
        output_path.mkdir(exist_ok=True, parents=True)
        # handle TOS
        if not self.tos_agreed(model_item, output_path):
            if not self.ask_tos(output_path):
                output_path.rmdir()
                raise Exception(" [!] You must agree to the terms of service to use this model.")
        logger.info("Downloading model to %s", output_path)
        try:
            if "fairseq" in model_name:
                self.download_fairseq_model(model_name, output_path)
            elif "github_rls_url" in model_item:
                self._download_github_model(model_item, output_path)
            elif "hf_url" in model_item:
                self._download_hf_model(model_item, output_path)

        except requests.RequestException as e:
            logger.exception("Failed to download the model file to %s", output_path)
            rmtree(output_path)
            raise e
        checkpoints = list(Path(output_path).glob("*.pt*"))
        if len(checkpoints) == 1:
            checkpoints[0].rename(checkpoints[0].parent / "model.pth")
        self.print_model_license(model_item=model_item)

    def check_if_configs_are_equal(self, model_name: str, model_item: ModelItem, output_path: Path) -> None:
        with fsspec.open(self._find_files(output_path)[1], "r", encoding="utf-8") as f:
            config_local = json.load(f)
        remote_url = None
        for url in model_item["hf_url"]:
            if "config.json" in url:
                remote_url = url
                break

        with fsspec.open(remote_url, "r", encoding="utf-8") as f:
            config_remote = json.load(f)

        if not config_local == config_remote:
            logger.info("%s is already downloaded however it has been changed. Redownloading it...", model_name)
            self.create_dir_and_download_model(model_name, model_item, output_path)

    def download_model(self, model_name: str) -> tuple[Path, Path | None, ModelItem]:
        """Download model files given the full model name.
        Model name is in the format
            'type/language/dataset/model'
            e.g. 'tts_model/en/ljspeech/tacotron'

        Every model must have the following files:
            - *.pth : pytorch model checkpoint file.
            - config.json : model config file.
            - scale_stats.npy (if exist): scale values for preprocessing.

        Args:
            model_name (str): model name as explained above.
        """
        model_item, model_full_name, model, md5sum = self._set_model_item(model_name)
        # set the model specific output path
        output_path = Path(self.output_prefix) / model_full_name
        if output_path.is_dir():
            if md5sum is not None:
                md5sum_file = output_path / "hash.md5"
                if md5sum_file.is_file():
                    with md5sum_file.open() as f:
                        if not f.read() == md5sum:
                            logger.info("%s has been updated, clearing model cache...", model_name)
                            self.create_dir_and_download_model(model_name, model_item, output_path)
                        else:
                            logger.info("%s is already downloaded.", model_name)
                else:
                    logger.info("%s has been updated, clearing model cache...", model_name)
                    self.create_dir_and_download_model(model_name, model_item, output_path)
            # if the configs are different, redownload it
            # ToDo: we need a better way to handle it
            if "xtts" in model_name:
                try:
                    self.check_if_configs_are_equal(model_name, model_item, output_path)
                except:
                    pass
            else:
                logger.info("%s is already downloaded.", model_name)
        else:
            self.create_dir_and_download_model(model_name, model_item, output_path)

        # find downloaded files
        output_model_path = output_path
        output_config_path = None
        if (
            model not in ["tortoise-v2", "bark", "knnvc"] and "fairseq" not in model_name and "xtts" not in model_name
        ):  # TODO:This is stupid but don't care for now.
            output_model_path, output_config_path = self._find_files(output_path)
        else:
            output_config_path = output_model_path / "config.json"
        if model == "knnvc" and not output_config_path.exists():
            knnvc_config = KNNVCConfig()
            knnvc_config.save_json(output_config_path)
        # update paths in the config.json
        self._update_paths(output_path, output_config_path)
        return output_model_path, output_config_path, model_item

    @staticmethod
    def _find_files(output_path: Path) -> tuple[Path, Path]:
        """Find the model and config files in the output path

        Args:
            output_path (str): path to the model files

        Returns:
            Tuple[str, str]: path to the model file and config file
        """
        model_file = None
        config_file = None
        for f in output_path.iterdir():
            if f.name in ["model_file.pth", "model_file.pth.tar", "model.pth", "checkpoint.pth"]:
                model_file = f
            elif f.name == "config.json":
                config_file = f
        if model_file is None:
            raise ValueError(" [!] Model file not found in the output path")
        if config_file is None:
            raise ValueError(" [!] Config file not found in the output path")
        return model_file, config_file

    @staticmethod
    def _find_speaker_encoder(output_path: Path) -> Path | None:
        """Find the speaker encoder file in the output path

        Args:
            output_path (str): path to the model files

        Returns:
            str: path to the speaker encoder file
        """
        speaker_encoder_file = None
        for f in output_path.iterdir():
            if f.name in ["model_se.pth", "model_se.pth.tar"]:
                speaker_encoder_file = f
        return speaker_encoder_file

    def _update_paths(self, output_path: Path, config_path: Path) -> None:
        """Update paths for certain files in config.json after download.

        Args:
            output_path (str): local path the model is downloaded to.
            config_path (str): local config.json path.
        """
        output_stats_path = output_path / "scale_stats.npy"
        output_d_vector_file_path = output_path / "speakers.json"
        output_d_vector_file_pth_path = output_path / "speakers.pth"
        output_speaker_ids_file_path = output_path / "speaker_ids.json"
        output_speaker_ids_file_pth_path = output_path / "speaker_ids.pth"
        speaker_encoder_config_path = output_path / "config_se.json"
        speaker_encoder_model_path = self._find_speaker_encoder(output_path)

        # update the scale_path.npy file path in the model config.json
        self._update_path("audio.stats_path", output_stats_path, config_path)

        # update the speakers.json file path in the model config.json to the current path
        self._update_path("d_vector_file", output_d_vector_file_path, config_path)
        self._update_path("d_vector_file", output_d_vector_file_pth_path, config_path)
        self._update_path("model_args.d_vector_file", output_d_vector_file_path, config_path)
        self._update_path("model_args.d_vector_file", output_d_vector_file_pth_path, config_path)

        # update the speaker_ids.json file path in the model config.json to the current path
        self._update_path("speakers_file", output_speaker_ids_file_path, config_path)
        self._update_path("speakers_file", output_speaker_ids_file_pth_path, config_path)
        self._update_path("model_args.speakers_file", output_speaker_ids_file_path, config_path)
        self._update_path("model_args.speakers_file", output_speaker_ids_file_pth_path, config_path)

        # update the speaker_encoder file path in the model config.json to the current path
        self._update_path("speaker_encoder_model_path", speaker_encoder_model_path, config_path)
        self._update_path("model_args.speaker_encoder_model_path", speaker_encoder_model_path, config_path)
        self._update_path("speaker_encoder_config_path", speaker_encoder_config_path, config_path)
        self._update_path("model_args.speaker_encoder_config_path", speaker_encoder_config_path, config_path)

    @staticmethod
    def _update_path(field_name: str, new_path: Path | None, config_path: Path) -> None:
        """Update the path in the model config.json for the current environment after download"""
        if new_path is not None and new_path.is_file():
            config = load_config(str(config_path))
            field_names = field_name.split(".")
            if len(field_names) > 1:
                # field name points to a sub-level field
                sub_conf = config
                for fd in field_names[:-1]:
                    if fd in sub_conf:
                        sub_conf = sub_conf[fd]
                    else:
                        return
                if isinstance(sub_conf[field_names[-1]], list):
                    sub_conf[field_names[-1]] = [new_path]
                else:
                    sub_conf[field_names[-1]] = new_path
            else:
                # field name points to a top-level field
                if field_name not in config:
                    return
                if isinstance(config[field_name], list):
                    config[field_name] = [new_path]
                else:
                    config[field_name] = new_path
            config.save_json(config_path)

    @staticmethod
    def _download_zip_file(file_url: str, output_folder: Path, progress_bar: bool) -> None:
        """Download the github releases"""
        # download the file
        r = requests.get(file_url, stream=True)
        # extract the file
        try:
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            ctx = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True) if progress_bar else nullcontext()
            temp_zip_name = output_folder / file_url.split("/")[-1]
            with open(temp_zip_name, "wb") as file, ctx as pbar:
                for data in r.iter_content(block_size):
                    if progress_bar:
                        pbar.update(len(data))
                    file.write(data)
            with zipfile.ZipFile(temp_zip_name) as z:
                z.extractall(output_folder)
            temp_zip_name.unlink()  # delete zip after extract
        except zipfile.BadZipFile:
            logger.exception("Bad zip file - %s", file_url)
            raise zipfile.BadZipFile  # pylint: disable=raise-missing-from
        # move the files to the outer path
        for file_path in z.namelist():
            src_path = output_folder / file_path
            if src_path.is_file():
                dst_path = output_folder / os.path.basename(file_path)
                if src_path != dst_path:
                    copyfile(src_path, dst_path)
        # remove redundant (hidden or not) folders
        for file_path in z.namelist():
            if (output_folder / file_path).is_dir():
                rmtree(output_folder / file_path)

    @staticmethod
    def _download_tar_file(file_url: str, output_folder: Path, progress_bar: bool) -> None:
        """Download the github releases"""
        # download the file
        r = requests.get(file_url, stream=True)
        # extract the file
        try:
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            ctx = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True) if progress_bar else nullcontext()
            temp_tar_name = output_folder / file_url.split("/")[-1]
            with open(temp_tar_name, "wb") as file, ctx as pbar:
                for data in r.iter_content(block_size):
                    if progress_bar:
                        pbar.update(len(data))
                    file.write(data)
            with tarfile.open(temp_tar_name) as t:
                t.extractall(output_folder)
                tar_names = t.getnames()
            temp_tar_name.unlink()  # delete tar after extract
        except tarfile.ReadError:
            logger.exception("Bad tar file - %s", file_url)
            raise tarfile.ReadError  # pylint: disable=raise-missing-from
        # move the files to the outer path
        for file_path in (output_folder / tar_names[0]).iterdir():
            src_path = file_path
            dst_path = output_folder / file_path.name
            if src_path != dst_path:
                copyfile(src_path, dst_path)
        # remove the extracted folder
        rmtree(output_folder / tar_names[0])

    @staticmethod
    def _download_model_files(file_urls: list[str], output_folder: str | os.PathLike[Any], progress_bar: bool) -> None:
        """Download the github releases"""
        output_folder = Path(output_folder)
        for file_url in file_urls:
            # download the file
            r = requests.get(file_url, stream=True)
            # extract the file
            base_filename = file_url.split("/")[-1]
            file_path = output_folder / base_filename
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            ctx = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True) if progress_bar else nullcontext()
            with open(file_path, "wb") as f, ctx as pbar:
                for data in r.iter_content(block_size):
                    if progress_bar:
                        pbar.update(len(data))
                    f.write(data)
