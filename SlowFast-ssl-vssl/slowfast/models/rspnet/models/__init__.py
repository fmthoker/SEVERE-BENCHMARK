"""
2020-05-09
Abstract the process of creating a model into get_model_class, model_class(num_classes=N)
"""

from pyhocon import ConfigTree, ConfigFactory
from torch import nn
import torch
from torch import nn
from typing import *
import logging

logger = logging.getLogger(__name__)


def get_model_class(**kwargs) -> Callable[[int], nn.Module]:
    """
    Pass the model config as parameters. For convinence, we change the cfg to dict, and then reverse it
    :param kwargs:
    :return:
    """
    logger.info(f'Using global get_model_class({kwargs})')

    cfg = ConfigFactory.from_dict(kwargs)

    arch: str = cfg.get_string('arch')

    if arch == 'resnet18':
        from .resnet import resnet18
        model_class = resnet18
    elif arch == 'resnet34':
        from .resnet import resnet34
        model_class = resnet34
    elif arch == 'resnet50':
        from .resnet import resnet50
        model_class = resnet50
    elif arch == 'torchvision-resnet18':
        from torchvision.models.video import r3d_18
        def model_class(num_classes):
            model = r3d_18(
                pretrained=cfg.get_bool('pretrained', default=False),
            )
            model.fc = nn.Linear(model.fc.in_features, num_classes, model.fc.bias is not None)
            return model
    elif arch == 'c3d':
        from .c3d import C3D
        model_class = C3D
    elif arch == 's3dg':
        from .s3dg import S3D_G
        model_class = S3D_G
    elif arch == 'mfnet':
        from .mfnet.mfnet_3d import MFNET_3D
        model_class = MFNET_3D
    elif arch == 'tsm':
        from models.tsm import TSM
        model_class = lambda num_classes=128: TSM(
            num_classes=num_classes,
            num_segments=cfg.get_int('num_segments'),
            base_model=cfg.get_string('base_model'),
            pretrain=cfg.get_string('pretrain', default=None),
        )
    elif arch.startswith('SLOWFAST'):
        from .slowfast import get_kineitcs_model_class_by_name
        model_class = get_kineitcs_model_class_by_name(arch)
    elif arch == 'r2plus1d-vcop':
        from .r2plus1d_vcop import R2Plus1DNet
        model_class = lambda num_classes=128: R2Plus1DNet(
            (1, 1, 1, 1),
            with_classifier=True,
            num_classes=num_classes
        )
    elif arch == 'r2plus1d_18':
        from .R2plus1D import R21D
        # num_classes does not matter
        model_class = lambda num_classes: R21D(
            with_classifier=False,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f'Unknown model architecture "{arch}"')

    return model_class


class ModelFactory:

    def __init__(self):
        pass

    def build_multitask_wrapper(self,arch, num_classes ):

        from slowfast.models.rspnet.moco.split_wrapper import MultiTaskWrapper

        model_cfg = {'arch':arch}
        model_class = get_model_class(**model_cfg)

        model = MultiTaskWrapper(model_class, num_classes=num_classes, finetune=True)

        return model
