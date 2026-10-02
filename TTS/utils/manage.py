import json
import logging
import os
import tarfile
import zipfile
from contextlib import nullcontext
from importlib import resources
from pathlib import Path
from shutil import copyfile, rmtree
from typing import Any, TypedDict

import requests
from tqdm import tqdm
from trainer.io import get_user_data_dir
from typing_extensions import Required

from TTS.config import load_config
from TTS.tts.configs.tortoise_config import TortoiseConfig
from TTS.vc.configs.knnvc_config import KNNVCConfig

logger = logging.getLogger(__name__)


class ModelItem(TypedDict, total=False):
    model_name: Required[str]
    model_type: Required[str]
    description: str
    license: str
    author: str
    contact: str
    tos_required: bool
    default_vocoder: str | None
    github_rls_url: str | list[str]
    repo_id: str
    allow: list[str] | None
    ignore: list[str] | None


LICENSE_URLS = {
    "cc by-nc-nd 4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "mpl": "https://www.mozilla.org/en-US/MPL/2.0/",
    "mpl2": "https://www.mozilla.org/en-US/MPL/2.0/",
    "mpl 2.0": "https://www.mozilla.org/en-US/MPL/2.0/",
    "mit": "https://choosealicense.com/licenses/mit/",
    "apache 2.0": "https://choosealicense.com/licenses/apache-2.0/",
    "apache2": "https://choosealicense.com/licenses/apache-2.0/",
    "cc-by-sa 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cpml": "https://tts-hub.github.io/cpml",
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
        f = resources.open_text("TTS", ".models.json") if models_file is None else open(models_file, encoding="utf-8")
        with f:
            self.models_dict = json.load(f)

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
                    hf = "*" if "repo_id" in self.models_dict[model_type][lang][dataset][model] else ""
                    logger.info(" %2d: %s/%s/%s/%s%s%s", model_count, model_type, lang, dataset, model, hf, downloaded)
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
        logger.info("(models marked with * are stored in the Hugging Face Hub cache folder)")
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

    def _download_hf_model(self, model_item: ModelItem) -> Path:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

        if not self.progress_bar:
            disable_progress_bars()
        output_path = snapshot_download(
            model_item["repo_id"],
            allow_patterns=model_item.get("allow"),
            ignore_patterns=model_item.get("ignore"),
        )
        if not self.progress_bar:
            enable_progress_bars()

        return Path(output_path)

    def download_fairseq_model(self, model_name: str, output_path: Path) -> None:
        URI_PREFIX = "https://dl.fbaipublicfiles.com/mms/tts/"
        _, lang, _, _ = model_name.split("/")
        model_download_uri = os.path.join(URI_PREFIX, f"{lang}.tar.gz")
        self._download_tar_file(model_download_uri, output_path, self.progress_bar)

    def _set_model_item(self, model_name: str) -> tuple[ModelItem, str, str]:
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
        else:
            # get model from models.json
            model_type, lang, dataset, model = model_name.split("/")
            model_item = self.models_dict[model_type][lang][dataset][model]
            model_item["model_type"] = model_type

        model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
        return model_item, model_full_name, model

    @staticmethod
    def ask_tos(model_full_path: Path) -> bool:
        """Ask the user to agree to the terms of service"""
        tos_path = model_full_path / "tos_agreed.txt"
        print("You must confirm the following:")
        print('  "I have purchased a commercial license from Coqui: licensing@coqui.ai"')
        print('  "Otherwise, I agree to the terms of the non-commercial CPML: https://tts-hub.github.io/cpml" - [y/n]')
        answer = input("   > ")
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

    def create_dir_and_download_model(self, model_name: str, model_item: ModelItem, output_path: Path) -> Path:
        output_path.mkdir(exist_ok=True, parents=True)
        # handle TOS
        if not self.tos_agreed(model_item, output_path):
            if not self.ask_tos(output_path):
                output_path.rmdir()
                raise RuntimeError(" [!] You must agree to the terms of service to use this model.")
        logger.info("Downloading model to %s", output_path)
        try:
            if "fairseq" in model_name:
                self.download_fairseq_model(model_name, output_path)
            elif "github_rls_url" in model_item:
                self._download_github_model(model_item, output_path)
            elif "repo_id" in model_item:
                output_path = self._download_hf_model(model_item)

        except requests.RequestException as e:
            logger.exception("Failed to download the model file to %s", output_path)
            rmtree(output_path)
            raise e
        self.print_model_license(model_item=model_item)
        return output_path

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
        model_item, model_full_name, model = self._set_model_item(model_name)
        # set the model specific output path
        output_path = self.output_prefix / model_full_name
        if output_path.is_dir() and "repo_id" not in model_item:
            logger.info("%s is already downloaded.", model_name)
        else:
            output_path = self.create_dir_and_download_model(model_name, model_item, output_path)
        # find downloaded files
        output_model_path = output_path
        output_config_path = output_model_path / "config.json"
        if model not in ["tortoise-v2", "bark", "knnvc"]:
            output_model_path, output_config_path = self._find_files(output_path)
        if model == "knnvc" and not output_config_path.exists():
            knnvc_config = KNNVCConfig()
            knnvc_config.save_json(output_config_path)
        if model == "tortoise-v2" and not output_config_path.exists():
            output_config_path = self.output_prefix / model_full_name / "config.json"
            if not output_config_path.is_file():
                tortoise_config = TortoiseConfig()
                tortoise_config.save_json(output_config_path)
        if all(x not in model_name for x in ("fairseq", "openvoice")):
            # Update paths in config, except for external models
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
        for f in output_path.iterdir():
            if f.name in ["model_file.pth", "model_file.pth.tar", "model.pth", "checkpoint.pth"]:
                model_file = f
            elif f.name == "config.json":
                config_file = f
        if model_file is None:
            checkpoints = list(output_path.rglob("*.pt*"))
            if len(checkpoints) == 1:
                model_file = checkpoints[0]
            else:
                raise ValueError(" [!] Model file not found in the output path")
        logger.debug("Found model checkpoint: %s", model_file)
        configs = list(output_path.rglob("config.json"))
        config_file = min(configs, key=lambda p: len(p.parts), default=None)
        if config_file is None:
            raise ValueError(" [!] Config file not found in the output path")
        logger.debug("Found config file: %s", config_file)
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
        config = load_config(config_path)
        has_changes = False

        def _set_path(field_name: str, new_path: Path) -> bool:
            """Set path in config if it differs from current value.

            Returns:
                bool: True if the value was changed, False otherwise.
            """
            keys = field_name.split(".")
            sub_conf = config
            for key in keys[:-1]:
                if key not in sub_conf:
                    return False
                sub_conf = sub_conf[key]
            if keys[-1] not in sub_conf:
                return False

            current_value = sub_conf[keys[-1]]

            if isinstance(current_value, list):
                current_value = current_value[0] if current_value else ""
                new_value = [new_path]
            else:
                new_value = new_path

            if current_value and Path(current_value) == new_path:
                return False
            sub_conf[keys[-1]] = new_value
            return True

        def _update_path(field_name: str, new_path: Path) -> bool:
            if not new_path.is_file():
                return False

            changed = _set_path(field_name, new_path)
            if not field_name.startswith("audio"):
                changed |= _set_path(f"model_args.{field_name}", new_path)
            return changed

        default_names = {
            "audio.stats_path": ["scale_stats.npy"],
            "d_vector_file": ["speakers.json", "speakers.pth"],
            "speakers_file": ["speaker_ids.json", "speaker_ids.pth"],
            "language_ids_file": ["language_ids.json"],
            "speaker_encoder_model_path": ["model_se.pth", "model_se.pth.tar"],
            "speaker_encoder_config_path": ["config_se.json"],
        }

        for field_name, defaults in default_names.items():
            name = config.get(field_name) or ""
            if isinstance(name, list):
                name = name[0]
            name = Path(name).name

            model_args_name = ""
            if config.get("model_args"):
                model_args_name = config["model_args"].get(field_name) or ""
                if isinstance(model_args_name, list):
                    model_args_name = model_args_name[0]
                model_args_name = Path(model_args_name).name

            if _update_path(field_name, output_path / name):
                has_changes = True
                continue
            elif _update_path(field_name, output_path / model_args_name):
                has_changes = True
                continue
            for default in defaults:
                if _update_path(field_name, output_path / default):
                    has_changes = True

        if has_changes:
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
                    if pbar:
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
                    if pbar:
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
                    if pbar:
                        pbar.update(len(data))
                    f.write(data)
