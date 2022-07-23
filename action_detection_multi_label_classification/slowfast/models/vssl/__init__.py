"""Base VSSL (video-ssl) backbone."""

from os.path import isfile

import torch
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.backbones import load_backbone

import warnings
warnings.filterwarnings("ignore")


class VSSL(nn.Module):
    """R(2+1)D Baseline"""

    def __init__(self, cfg):
        super(VSSL, self).__init__()
        self.num_pathways = 1
        self.enable_detection = cfg.DETECTION.ENABLE
        self._construct_network(cfg)

    def _construct_network(self, cfg):

        # define the encoder
        print("\nConstructing VSSL backbone...")
        print(f"Initializing backbone by {cfg.MODEL.INIT_METHOD}...\n")
        backbone = load_backbone(
            backbone=cfg.MODEL.ARCH,
            init_method=cfg.MODEL.INIT_METHOD,
            ckpt_path=cfg.MODEL.get("CKPT", None),
        )
        # only keep the relevant layers (HARDCODED w.r.t. R2PLUS1D-18)
        self.encoder = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

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
    args.cfg_file = join(abspath(__file__), "../../../../configs/Charades/VSSL/32x8_112x112_R18_supervised.yaml")
    cfg = load_config(args)

    ## check only classification
    print(":::: Test without detection on Charades ::::")

    args.cfg_file = join(abspath(__file__), "../../../../configs/Charades/VSSL/32x8_112x112_R18_supervised.yaml")
    cfg = load_config(args)

    cfg.DETECTION.ENABLE = False
    cfg.DATA.NUM_FRAMES = 64

    # load model
    model = VSSL(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    # E = model.encoder
    # z = E(x)
    # print(E.layer4(E.layer3(E.layer2(E.layer1(E.stem(x))))).shape)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print("Test passed!\n")

    cfg.DATA.NUM_FRAMES = 1

    # load model
    model = VSSL(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print(f"Test passed!\n")

    cfg.DATA.NUM_FRAMES = 8

    # load model
    model = VSSL(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print(f"Test passed!\n")
