"""Implemented MOCO with R2+1D backbone."""

from os.path import isfile

import torch
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.models.r2plus1d import video_resnet


class MOCO(nn.Module):
    """MOCO Baseline"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(MOCO, self).__init__()
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
        self.init_weights(self.encoder, cfg.MODEL.CKPT)

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

    def init_weights(self, encoder, ckpt_path=None):
        # load from pre-trained, before DistributedDataParallel constructor
        if ckpt_path is not None:
            if isfile(ckpt_path):
                print("=> loading checkpoint '{}'".format(ckpt_path))
                checkpoint = torch.load(ckpt_path, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                #args.start_epoch = 0
                msg = encoder.load_state_dict(state_dict, strict=False)
                print(msg)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

                print("=> loaded pre-trained model '{}'".format(ckpt_path))
            else:
                print("=> no checkpoint found at '{}'".format(ckpt_path))

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
    # args.cfg_file = join(abspath(__file__), "../../../../configs/AVA/MOCO/diva_32x2_112x112_R18_v2.2.yaml")
    # cfg = load_config(args)

    # # set number of frames
    # cfg.DATA.NUM_FRAMES = 32

    # # load model
    # model = MOCO(cfg)

    # x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    # # 5 boxes for the 1st sample
    # boxes = torch.randn(5, 4)
    # boxes = torch.hstack([torch.zeros(5).view((-1, 1)), boxes])
    # y = model([x], boxes)
    # assert y.shape == torch.Size([5, 80])

    ## check only classification
    print(":::: Test without detection on Charades ::::")

    args.cfg_file = join(abspath(__file__), "../../../../configs/Charades/MOCO/das6_32x8_112x112_R18.yaml")
    cfg = load_config(args)

    cfg.DETECTION.ENABLE = False
    cfg.DATA.NUM_FRAMES = 64

    # load model
    model = MOCO(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print("Test passed!")

