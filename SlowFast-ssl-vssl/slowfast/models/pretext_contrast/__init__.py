"""Defines PretextContrast network."""
from os import stat
import re
from os.path import isfile

import torch
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.models.r2plus1d import video_resnet
# from slowfast.models.pretext_contrast.network import R21D


def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path, map_location="cpu")
    for name, params in pretrained_weights.items():
        if 'module' in name:
            name = name[name.find('module')+7:]
            adjusted_weights[name] = params
            print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights


class PretextContrast(nn.Module):
    """PretextContrast Baseline"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PretextContrast, self).__init__()
        self.num_pathways = 1
        self.enable_detection = cfg.DETECTION.ENABLE
        self._construct_network(cfg)
        self.init_weights(cfg.MODEL.CKPT)

    def _construct_network(self, cfg):
        """
        PretextContrast

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # define the encoder
        self.encoder = video_resnet.__dict__[cfg.MODEL.ARCH](pretrained=False)

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
    
    @staticmethod
    def find_key(pattern, query, values):
        """Returns all strings in queries (list of strings) which have the pattern."""
        return [x for x in values if re.search(pattern, x)]
    
    @staticmethod
    def modify_key(key):
        start_char = key.split(".")[0]
        if start_char.isdigit():

            if int(start_char) == 0:
                key = key.replace("0.", "stem.", 1)

            if int(start_char) > 0:
                key = key.replace(f"{start_char}.", f"layer{start_char}.", 1)

        return key
    
    def init_weights(self, ckpt_path):
        assert isfile(ckpt_path), f"Checkpoint does not exist at {ckpt_path}."
        # pretrained_weights = load_pretrained_weights(ckpt_path)
        pretrained_weights = torch.load(ckpt_path, map_location="cpu")
        pretrained_weights = {k.replace("module.base_network.", ""):v for k, v in pretrained_weights.items()}
        pretrained_weights = {self.modify_key(k):v for k, v in pretrained_weights.items()}

        pt_keys = list(pretrained_weights.keys())
        en_keys = list(self.encoder.state_dict().keys())

        intersection = set(pretrained_weights).intersection(set(en_keys))
        print(f"\n::::: Found intersection of {len(intersection)} keys between checkpoint and encoder. \n")
        
        msg = self.encoder.load_state_dict(pretrained_weights, strict=False)
        print(msg)
        print(f"\n::::: Loaded pretrained weights from {ckpt_path} \n")


if __name__ == "__main__":
    from tools.run_net import parse_args, load_config
    from os.path import join, abspath

    # load cfg
    args = parse_args()
    # args.cfg_file = join(abspath(__file__), "../../../../configs/AVA/PRETEXT_CONTRAST/diva_32x2_112x112_R18_v2.2.yaml")
    # cfg = load_config(args)

    # # load model
    # model = PretextContrast(cfg)

    # # test with sample inputs
    # x = torch.randn(1, 3, 32, 112, 112)
    # # 5 boxes for the 1st sample
    # boxes = torch.randn(5, 4)
    # boxes = torch.hstack([torch.zeros(5).view((-1, 1)), boxes])
    # y = model([x], boxes)

    # assert y.shape == torch.Size([5, 80])

    ## check only classification
    print(":::: Test without detection on Charades ::::")

    args.cfg_file = join(abspath(__file__), "../../../../configs/Charades/PRETEXT_CONTRAST/das6_32x8_112x112_R18.yaml")
    cfg = load_config(args)

    cfg.DETECTION.ENABLE = False
    cfg.DATA.NUM_FRAMES = 64

    # load model
    model = PretextContrast(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print("Test passed!")
