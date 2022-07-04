"""Defines the RSPNet model."""

from os.path import isfile

import torch
from torch._C import Value
import torch.nn as nn
import torchvision.models.video as models

from slowfast.models import head_helper
from slowfast.models.r2plus1d import video_resnet
from slowfast.models.rspnet.models import ModelFactory, get_model_class
from slowfast.models.rspnet.models.r2plus1d_vcop import R2Plus1DNet
from slowfast.models.r2plus1d import video_resnet


class RSPNet(nn.Module):
    """RSPNet Baseline"""

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(RSPNet, self).__init__()
        self.num_pathways = 1
        self.enable_detection = cfg.DETECTION.ENABLE
        self._construct_network(cfg)
        self.init_weights(ckpt_path=cfg.MODEL.CKPT)

    def _construct_network(self, cfg):
        """
        R(2 + 1)D

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # define the encoder
        if cfg.MODEL.ARCH == "r2plus1d_18":
            # self.encoder = R2Plus1DNet(
            #     layer_sizes=(1, 1, 1, 1), with_classifier=False,
            # )
            # self.encoder = video_resnet.__dict__[cfg.MODEL.ARCH](pretrained=False)

            #rspnet imports
            from slowfast.models.rspnet.models.R2plus1D import R21D
            # from slowfast.models.rspnet.models import ModelFactory

            # rspnet model
            # model_factory = ModelFactory()
            # self.encoder = model_factory.build_multitask_wrapper("r2plus1d_18", num_classes=101)
            self.encoder = R21D(num_classes=None, with_classifier=False)
            
        else:
            raise ValueError(f"cfg.MODEL.ARCH = {cfg.MODEL.ARCH} currently not supported for RSPNet")

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
    
    def init_weights(self, ckpt_path):
        cp = torch.load(ckpt_path, map_location='cpu')

        if 'model' in cp and 'arch' in cp:
            print('Loading MoCo checkpoint from %s (epoch %d)', ckpt_path, cp['epoch'])
            moco_state = cp['model']
            prefix = 'encoder_q.'
        else:
            # This checkpoint is from third-party
            #logger.info('Loading third-party model from %s', ckpt_path)
            print('Loading third-party model from %s', ckpt_path)
            if 'state_dict' in cp:
                moco_state = cp['state_dict']
            else:
                # For c3d
                moco_state = cp
                #logger.warning('if you are not using c3d sport1m, maybe you use wrong checkpoint')
                print('if you are not using c3d sport1m, maybe you use wrong checkpoint')
            if next(iter(moco_state.keys())).startswith('module'):
                prefix = 'module.'
            else:
                prefix = ''

        """
        fc -> fc. for c3d sport1m. Beacuse fc6 and fc7 is in use.
        """
        blacklist = ['fc.', 'linear', 'head', 'new_fc', 'fc8']
        blacklist += ['encoder_fuse']

        def filter(k):
            return k.startswith(prefix) and not any(k.startswith(f'{prefix}{fc}') for fc in blacklist)

        model_state = {k[len(prefix):]: v for k, v in moco_state.items() if filter(k)}
        model_state = {k.replace("encoder.", ""):v for k, v in model_state.items()}
        msg = self.encoder.load_state_dict(model_state, strict=False)
        print(f'Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}')


if __name__ == "__main__":
    from tools.run_net import parse_args, load_config

    # load cfg
    from os.path import join, abspath
    args = parse_args()
    args.cfg_file = join(abspath(__file__), "../../../../configs/AVA/RSPNET/diva_32x2_112x112_R18_v2.2.yaml")
    cfg = load_config(args)

    # # load model
    # model = RSPNet(cfg)

    # # test with sample inputs
    # x = torch.randn(1, 3, 32, 112, 112)

    # # 5 boxes for the 1st sample
    # boxes = torch.randn(5, 4)
    # boxes = torch.hstack([torch.zeros(5).view((-1, 1)), boxes])
    # y = model([x], boxes)
    # assert y.shape == torch.Size([5, 80])

    # E = model.encoder
    # z = E(x)
    # assert z.shape == torch.Size([1, 512, 4, 7, 7])

    ## check only classification
    print(":::: Test without detection on Charades ::::")

    args.cfg_file = join(abspath(__file__), "../../../../configs/Charades/RSPNET/das6_32x8_112x112_R18.yaml")
    cfg = load_config(args)

    cfg.DETECTION.ENABLE = False
    cfg.DATA.NUM_FRAMES = 64

    # load model
    model = RSPNet(cfg)
    x = torch.randn(1, 3, cfg.DATA.NUM_FRAMES, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print("Test passed!")
