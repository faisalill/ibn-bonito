import os
import sys
import numpy as np
from tqdm import tqdm
from time import perf_counter
from functools import partial
from datetime import timedelta
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import torch
from torch.nn import Module
from torch.nn.init import orthogonal_
from torch.nn.utils.fusion import fuse_conv_bn_eval
from pathlib import Path
import toml
from glob import glob
import re
from importlib import import_module
from torch.cuda import get_device_capability

__dir__ = Path(__file__).parent
__models_dir__ = __dir__ / "models"
__data_dir__ = __dir__ / "data"


def half_supported():
    """
    Returns whether FP16 is support on the GPU
    """
    try:
        return get_device_capability()[0] >= 7
    except:
        return False


def load_symbol(config, symbol):
    """
    Dynamic load a symbol from module specified in model config.
    """
    if not isinstance(config, dict):
        if not os.path.isdir(config) and os.path.isdir(
            os.path.join(__models_dir__, config)
        ):
            dirname = os.path.join(__models_dir__, config)
        else:
            dirname = config
        config = toml.load(os.path.join(dirname, "config.toml"))
    imported = import_module(config["model"]["package"])
    return getattr(imported, symbol)


def match_names(state_dict, model):
    keys_and_shapes = lambda state_dict: zip(
        *[
            (k, s)
            for s, i, k in sorted(
                [(v.shape, i, k) for i, (k, v) in enumerate(state_dict.items())]
            )
        ]
    )
    k1, s1 = keys_and_shapes(state_dict)
    k2, s2 = keys_and_shapes(model.state_dict())
    assert s1 == s2
    remap = dict(zip(k1, k2))
    return OrderedDict([(k, remap[k]) for k in state_dict.keys()])


def get_last_checkpoint(dirname):
    weight_files = glob(os.path.join(dirname, "weights_*.tar"))
    if not weight_files:
        raise FileNotFoundError("no model weights found in '%s'" % dirname)
    weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])
    return os.path.join(dirname, "weights_%s.tar" % weights)


def set_config_defaults(
    config, chunksize=None, batchsize=None, overlap=None, quantize=False
):
    basecall_params = config.get("basecaller", {})
    # use `value or dict.get(key)` rather than `dict.get(key, value)` to make
    # flags override values in config
    basecall_params["chunksize"] = chunksize or basecall_params.get("chunksize", 4000)
    basecall_params["overlap"] = (
        overlap if overlap is not None else basecall_params.get("overlap", 500)
    )
    basecall_params["batchsize"] = batchsize or basecall_params.get("batchsize", 64)
    basecall_params["quantize"] = (
        basecall_params.get("quantize") if quantize is None else quantize
    )
    config["basecaller"] = basecall_params
    return config


def load_model(
    dirname,
    device,
    weights=None,
    half=None,
    chunksize=None,
    batchsize=None,
    overlap=None,
    quantize=False,
    use_koi=False,
):
    """
    Load a model config and weights off disk from `dirname`.
    """
    if not os.path.isdir(dirname) and os.path.isdir(
        os.path.join(__models_dir__, dirname)
    ):
        dirname = os.path.join(__models_dir__, dirname)
    weights = (
        get_last_checkpoint(dirname)
        if weights is None
        else os.path.join(dirname, "weights_%s.tar" % weights)
    )
    config = toml.load(os.path.join(dirname, "config.toml"))
    config = set_config_defaults(config, chunksize, batchsize, overlap, quantize)
    return _load_model(weights, config, device, half, use_koi)


def _load_model(model_file, config, device, half=None, use_koi=False):
    device = torch.device(device)
    Model = load_symbol(config, "Model")
    model = Model(config)

    if use_koi:
        config["basecaller"]["chunksize"] -= (
            config["basecaller"]["chunksize"] % model.stride
        )
        # overlap must be even multiple of stride for correct stitching
        config["basecaller"]["overlap"] -= config["basecaller"]["overlap"] % (
            model.stride * 2
        )
        model.use_koi(
            batchsize=config["basecaller"]["batchsize"],
            chunksize=config["basecaller"]["chunksize"],
            quantize=config["basecaller"]["quantize"],
        )

    state_dict = torch.load(model_file, map_location=device)
    state_dict = {
        k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()
    }
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if half is None:
        half = half_supported()

    if half:
        model = model.half()
    model.eval()
    model.to(device)
    return model


def health():
    dirname = "./models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0/"
    weights = get_last_checkpoint(dirname)
    print("Weight Loaded Successfully")
    config = toml.load(os.path.join(dirname, "config.toml"))
    chunksize = config["basecaller"]["chunksize"]
    batchsize = config["basecaller"]["batchsize"]
    overlap = config["basecaller"]["overlap"]
    quantize = False
    config = set_config_defaults(config, chunksize, batchsize, overlap, quantize)
    _load_model(weights, config, "cpu", use_koi=False)
    print("Code is not breaking!!!")


health()
