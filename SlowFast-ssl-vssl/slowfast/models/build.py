#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

from slowfast.models.r2plus1d import R2Plus1D
MODEL_REGISTRY._do_register("R2Plus1D", R2Plus1D)

from slowfast.models.ctp import CTP
MODEL_REGISTRY._do_register("CTP", CTP)

from slowfast.models.avid_cma import AVID_CMA
MODEL_REGISTRY._do_register("AVID_CMA", AVID_CMA)

from slowfast.models.gdt import GDTBase
MODEL_REGISTRY._do_register("GDTBase", GDTBase)

from slowfast.models.pretext_contrast import PretextContrast
MODEL_REGISTRY._do_register("PretextContrast", PretextContrast)

from slowfast.models.rspnet import RSPNet
MODEL_REGISTRY._do_register("RSPNet", RSPNet)

from slowfast.models.selavi import SELAVI
MODEL_REGISTRY._do_register("SELAVI", SELAVI)

from slowfast.models.tclr import TCLR
MODEL_REGISTRY._do_register("TCLR", TCLR)

from slowfast.models.videomoco import VideoMoCo
MODEL_REGISTRY._do_register("VideoMoCo", VideoMoCo)

from slowfast.models.moco import MOCO
MODEL_REGISTRY._do_register("MOCO", MOCO)

from slowfast.models.vssl import VSSL
MODEL_REGISTRY._do_register("VSSL", VSSL)

# single script to build all models
from slowfast.backbones import load_backbone


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    # model = load_backbone(
    #     backbone="r2plus1d_18",
    #     init_method=name,
    #     ckpt_path=cfg.MODEL.get("CKPT", None),
    # )

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True,
        )
    return model
