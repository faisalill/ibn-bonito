import os
import shutil
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from logging import getLogger
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

__dir__ = Path(__file__).parent
__models_dir__ = __dir__ / "models"
__data_dir__ = __dir__ / "data"

logger = getLogger("bonito")


def tqdm_environ():
    """Get tqdm settings from environment variables"""
    kwargs = {}
    try:
        interval = os.getenv("BONITO_PBAR_INTERVAL", None)
        if interval is not None:
            kwargs.update(
                dict(mininterval=float(interval), maxinterval=float(interval))
            )
    except ValueError as exc:
        logger.warning(f"Couldn't parse BONITO_PBAR_INTERVAL as float - {exc}")

    try:
        disable = os.getenv("BONITO_PBAR_DISABLE", None)
        if disable is not None:
            kwargs.update(dict(disable=bool(int(disable))))
    except ValueError as exc:
        logger.warning(f"couldn't parse BONITO_PBAR_DISABLE as bool - {exc}")

    return kwargs


class Printer:
    def show(self, model_list):
        print("[available models]")
        for model in model_list:
            print(f" - {model}")

    def show_specific(self, model_name):
        print("Downloading model:")
        print(f" - {model_name}", file=sys.stderr)


class Downloader:
    """
    Small class for downloading models and training assets.
    """

    __url__ = "https://cdn.oxfordnanoportal.com/software/analysis/bonito"

    def __init__(self, out_dir: Path, force=False):
        print(f"[Downloading to {out_dir}]", file=sys.stderr)
        out_dir.mkdir(exist_ok=True, parents=True)
        self.path = out_dir
        self.force = force

    def download(self, fname):
        url = f"{self.__url__}/{fname}.zip"
        fpath = self.path / f"{fname}"
        fpath_zip = self.path / f"{fname}.zip"

        if fpath.exists():
            if self.force:
                fpath.unlink() if fpath.is_file() else shutil.rmtree(fpath)
            else:
                print(f" - Skipping: {fname}", file=sys.stderr)
                return fpath

        # create the requests for the file
        req = requests.get(url, stream=True)
        total = int(req.headers.get("content-length", 0))

        # download the file content
        with tqdm(
            total=total,
            unit="iB",
            ascii=True,
            ncols=100,
            unit_scale=True,
            leave=False,
            **tqdm_environ(),
        ) as t:
            with fpath_zip.open("wb") as f:
                for data in req.iter_content(1024):
                    if b"Error" in data:
                        raise FileNotFoundError(
                            f" - Failed to download: {fname}\n{data.decode()}"
                        )
                    f.write(data)
                    t.update(len(data))
        self._unzip(fpath_zip)
        print(f" - Downloaded: {fname}", file=sys.stderr)
        return fpath

    def _unzip(self, fpath):
        unzip_path = fpath.parent.with_suffix("")
        with ZipFile(fpath, "r") as zfile:
            zfile.extractall(path=unzip_path)
        fpath.unlink()
        return unzip_path


models = [
    "dna_r10.4.1_e8.2_400bps_fast@v5.0.0",
    "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
    "dna_r10.4.1_e8.2_400bps_sup@v5.0.0",
    "dna_r10.4.1_e8.2_400bps_fast@v4.3.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.3.0",
    "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
    "dna_r10.4.1_e8.2_400bps_fast@v4.2.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
    "dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
    "dna_r10.4.1_e8.2_260bps_fast@v4.1.0",
    "dna_r10.4.1_e8.2_260bps_hac@v4.1.0",
    "dna_r10.4.1_e8.2_260bps_sup@v4.1.0",
    "dna_r10.4.1_e8.2_400bps_fast@v4.1.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.1.0",
    "dna_r10.4.1_e8.2_400bps_sup@v4.1.0",
    "dna_r10.4.1_e8.2_260bps_fast@v4.0.0",
    "dna_r10.4.1_e8.2_260bps_hac@v4.0.0",
    "dna_r10.4.1_e8.2_260bps_sup@v4.0.0",
    "dna_r10.4.1_e8.2_400bps_fast@v4.0.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
    "dna_r10.4.1_e8.2_400bps_sup@v4.0.0",
    "dna_r10.4.1_e8.2_260bps_fast@v3.5.2",
    "dna_r10.4.1_e8.2_260bps_hac@v3.5.2",
    "dna_r10.4.1_e8.2_260bps_sup@v3.5.2",
    "dna_r10.4.1_e8.2_400bps_fast@v3.5.2",
    "dna_r10.4.1_e8.2_400bps_hac@v3.5.2",
    "dna_r10.4.1_e8.2_400bps_sup@v3.5.2",
    "dna_r9.4.1_e8_sup@v3.3",
    "dna_r9.4.1_e8_hac@v3.3",
    "dna_r9.4.1_e8_fast@v3.4",
    "rna004_130bps_fast@v5.0.0",
    "rna004_130bps_hac@v5.0.0",
    "rna004_130bps_sup@v5.0.0",
    "rna004_130bps_fast@v3.0.1",
    "rna004_130bps_hac@v3.0.1",
    "rna004_130bps_sup@v3.0.1",
    "rna002_70bps_fast@v3",
    "rna002_70bps_hac@v3",
    "rna002_70bps_sup@v3",
]

training_data_sets = [
    "example_data_dna_r9.4.1_v0",
    "example_data_dna_r10.4.1_v0",
    "example_data_rna004_v0",
]


# bonito download --models --show
def download_models_show():
    pr = Printer()
    pr.show(models)


# bonito download --models {model_name}
def download_model_specific(model_name):
    if model_name not in models:
        print("Model does not exist. Please pick a model from:")
        pr = Printer()
        pr.show(models)
    else:
        pr = Printer()
        pr.show_specific(model_name)
        dl = Downloader(__models_dir__, True)
        dl.download(model_name)


# bonito download --models
def download_models_all(model_list):
    dl = Downloader(__models_dir__, True)
    for model in model_list:
        if model not in models:
            print(f"{model} does not exist. Please pick a model from:")
            pr = Printer()
            pr.show(models)
            break
        else:
            print(f"Downloading Model: {model}")
            dl.download(model)

    print("Downloading All Models: ")


def download_training_show_all():
    print("[available training data sets]")
    for set in training_data_sets:
        print(f" - {set}")


# bonito download --training {training_data_set_name}
def download_training_data_specific(training_data_set_name):
    if training_data_set_name not in training_data_sets:
        print(
            f"{training_data_set_name} does not exist. Please pick a set from the following: "
        )
        print("[available training sets]")
        for set in training_data_sets:
            print(f" - {set}")
    else:
        dl = Downloader(__data_dir__, True)
        print(f"Downloading Training Data: {training_data_set_name}")
        dl.download(training_data_set_name)
        print("Downloaded Training Data Set Successfully")


def main():
    # download_models_show()
    # download_model_specific("dna_r10.4.1_e8.2_400bps_hac@v5.0.0")
    ###### download_models_all(models)
    # download_training_show_all()
    download_training_data_specific("example_data_dna_r10.4.1_v0")
    print("download.py")


main()
