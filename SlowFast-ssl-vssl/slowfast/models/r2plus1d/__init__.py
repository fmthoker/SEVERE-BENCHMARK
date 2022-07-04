"""Implemented R(2 + 1)D network."""

from os.path import isfile

import torch
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.models.r2plus1d import video_resnet


class R2Plus1D(nn.Module):
    """R(2+1)D Baseline"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(R2Plus1D, self).__init__()
        self.num_pathways = 1
        self.enable_detection = cfg.DETECTION.ENABLE
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        R(2 + 1)D

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # define the encoder
        pretrained = cfg.MODEL.get("PRETRAINED", True)
        self.encoder = video_resnet.__dict__[cfg.MODEL.ARCH](pretrained=pretrained)

        # temporary hardcoding
        # Temporal pooling: [n_frames // 8, 1, 1]
        # this is because R2+1D18 reduces the T dimension by a factor of 8
        # if number of frames < 8, then the temporal dimension is 1
        temporal_pool = max(cfg.DATA.NUM_FRAMES // 8, 1)
        # if cfg.DATA.NUM_FRAMES // 8 >= 1:
        #     temporal_pool = cfg.DATA.NUM_FRAMES // 8
        # else:
        #     temporal_pool = 1
        print(":::: Using temporal pooling: {}".format(temporal_pool))

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[512],
                num_classes=cfg.MODEL.NUM_CLASSES,
                # pool_size=[[cfg.DATA.NUM_FRAMES // 8, 1, 1]],
                pool_size=[[temporal_pool, 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[512],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[temporal_pool, 7, 7]],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):

        for pathway in range(self.num_pathways):
            x[pathway] = self.encoder(x[pathway])
        
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)

        return x

    def freeze_fn(self, freeze_mode):

        if freeze_mode == 'bn_parameters':
            print("Freezing all BN layers\' parameters.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    # shutdown parameters update in frozen mode
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        elif freeze_mode == 'bn_statistics':
            print("Freezing all BN layers\' statistics.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    # shutdown running statistics update in frozen mode
                    m.eval()


if __name__ == "__main__":
    
    from tools.run_net import parse_args, load_config

    # load cfg
    from os.path import join, abspath
    args = parse_args()
    args.cfg_file = join(abspath(__file__), "../../../../configs/AVA/R2PLUS1D/32x2_112x112_R18_v2.2.yaml")
    cfg = load_config(args)
    
    ## check with detection
    print()
    print(":::: Test with detection on AVA ::::")
    cfg.DETECTION.ENABLE = True

    # set number of frames
    cfg.DATA.NUM_FRAMES = 32

    # load model
    model = R2Plus1D(cfg)

    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    # 5 boxes for the 1st sample
    boxes = torch.randn(5, 4)
    boxes = torch.hstack([torch.zeros(5).view((-1, 1)), boxes])
    y = model([x], boxes)
    assert y.shape == torch.Size([5, cfg.MODEL.NUM_CLASSES])
    print("Test passed!\n")

    # set number of frames
    cfg.DATA.NUM_FRAMES = 16

    # load model
    model = R2Plus1D(cfg)

    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 256, 256)
    # 5 boxes for the 1st sample
    boxes = torch.randn(5, 4)
    boxes = torch.hstack([torch.zeros(5).view((-1, 1)), boxes])
    y = model([x], boxes)
    assert y.shape == torch.Size([5, cfg.MODEL.NUM_CLASSES])
    print("Test passed!\n")

    # set number of frames
    cfg.DATA.NUM_FRAMES = 2

    # load model
    model = R2Plus1D(cfg)

    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    # 5 boxes for the 1st sample
    boxes = torch.randn(5, 4)
    boxes = torch.hstack([torch.zeros(5).view((-1, 1)), boxes])
    y = model([x], boxes)
    assert y.shape == torch.Size([5, cfg.MODEL.NUM_CLASSES])
    print("Test passed!\n")


    ## check only classification
    print(":::: Test without detection on Charades ::::")

    args.cfg_file = join(abspath(__file__), "../../../../configs/Charades/R2PLUS1D/64x2_112x112_R18.yaml")
    cfg = load_config(args)

    cfg.DETECTION.ENABLE = False
    cfg.DATA.NUM_FRAMES = 64

    # load model
    model = R2Plus1D(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print("Test passed!\n")

    cfg.DATA.NUM_FRAMES = 1

    # load model
    model = R2Plus1D(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print(f"Test passed!\n")

    cfg.DATA.NUM_FRAMES = 8

    # load model
    model = R2Plus1D(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print(f"Test passed!\n")
