# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from genericpath import exists
# from slowfast.models.avid_cma.video import *
from slowfast.models.avid_cma.video_resnet import r2plus1d_18 as PyTorchR2Plus1D
# from slowfast.models.avid_cma.audio import *
# from slowfast.models.avid_cma.av_wrapper import *

from slowfast.models import head_helper



class AVID_CMA(nn.Module):
    """R(2+1)D Baseline"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(AVID_CMA, self).__init__()
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
        self.encoder = PyTorchR2Plus1D(pretrained=False)
        self.init_weights_from_ckpt(ckpt_path=cfg.MODEL.CKPT)

        # temporary hardcoding
        # Temporal pooling: [n_frames // 8, 1, 1]
        # this is because R2+1D18 reduces the T dimension by a factor of 8
        assert cfg.DATA.NUM_FRAMES > 8, "Temporal pooling requires NUM_FRAMES > 8.\
            Current NUM_FRAMES = {}".format(cfg.DATA.NUM_FRAMES)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[512],
                num_classes=cfg.MODEL.NUM_CLASSES,
                # pool_size=[[cfg.DATA.NUM_FRAMES // 8, 1, 1]],
                pool_size=[[cfg.DATA.NUM_FRAMES // 8, 1, 1]],
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
                pool_size=[[cfg.DATA.NUM_FRAMES // 8, 7, 7]],
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
    
    def init_weights_from_ckpt(self, ckpt_path):
        """Loads weights from checkpoint."""
        assert exists(ckpt_path), "Cannot find checkpoint at {}".format(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(":::: Loading weights from {}".format(ckpt_path))
        # only need to replace the keys corresponding to the video model
        msg = self.encoder.load_state_dict(
            {k.replace('module.video_model.', ''): ckpt['model'][k] for k in ckpt['model']},
            strict=False,
        )
        print(":::: Checkpoint loaded with {}".format(msg))

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
    import torch
    from os.path import join, abspath
    from tools.run_net import parse_args, load_config

    # load cfg
    args = parse_args()
    # args.cfg_file = join(abspath(__file__), "../../../../configs/AVA/AVID_CMA/diva_32x2_112x112_R18_v2.2.yaml")
    # cfg = load_config(args)

    # # load model
    # model = AVID_CMA(cfg)

    # # test with sample inputs
    # x = torch.randn(1, 3, 32, 112, 112)
    # # 5 boxes for the 1st sample
    # boxes = torch.randn(5, 4)
    # boxes = torch.hstack([torch.zeros(5).view((-1, 1)), boxes])
    # y = model([x], boxes)
    # assert y.shape == torch.Size([5, 80])
    
    ## check only classification
    print(":::: Test without detection on Charades ::::")

    args.cfg_file = join(abspath(__file__), "../../../../configs/Charades/AVID_CMA/das6_32x8_112x112_R18.yaml")
    cfg = load_config(args)

    cfg.DETECTION.ENABLE = False
    cfg.DATA.NUM_FRAMES = 64

    # load model
    model = AVID_CMA(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print("Test passed!")

