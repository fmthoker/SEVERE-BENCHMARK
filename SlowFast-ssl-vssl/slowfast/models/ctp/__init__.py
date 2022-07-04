
import torch
from torch import nn
import torch.distributed as dist

# import utils.logger
# from utils import main_utils
import yaml
import os


from slowfast.models.ctp.r3d import R3D, R2Plus1D
from slowfast.models import head_helper


arch_options = ["R3D", "R2Plus1D"]


class CTP(nn.Module):
    """
    CTP model.
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(CTP, self).__init__()
        self.cfg = cfg
        self.num_pathways = 1
        self.enable_detection = cfg.DETECTION.ENABLE
        self._construct_network(cfg)
        # self._init_weights(self.encoder, ckpt_path=cfg.MODEL.CKPT)
    
    def _init_weights(self, encoder, ckpt_path):

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint['state_dict']

            for q,k in zip(list(encoder.state_dict().keys()),list(state_dict.keys())):
                # retain only encoder_q up to before the embedding layer
                #print(q,k[len("backbone."):])
                if k.startswith('backbone.'):# and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
                    #state_dict[q] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = encoder.load_state_dict(state_dict, strict=False)
            print(f":::::: Initilized weight from {ckpt_path} with the following information \n {msg} \n")
        else:
            print(":::::: Initializing weights randomly since cfg.MODEL.CKPT = None.")

    def _construct_network(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        # define encoder
        args = dict(
            depth=18,
            num_class=None,
            num_stages=4,
            stem=dict(
                temporal_kernel_size=3,
                temporal_stride=1,
                in_channels=3,
                with_pool=False,
            ),
            down_sampling=[False, True, True, True],
            channel_multiplier=1.0,
            bottleneck_multiplier=1.0,
            with_bn=True,
            zero_init_residual=False,
            pretrained=None,
        )
        arch = cfg.MODEL.ARCH
        assert arch in arch_options
        self.encoder = eval(arch)(**args)
        self._init_weights(self.encoder, ckpt_path=cfg.MODEL.CKPT)

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


if __name__ == "__main__":
    from os.path import join, abspath
    from tools.run_net import parse_args, load_config

    print("::: Testing detction with AVA ...")
    # load cfg
    args = parse_args()
    args.cfg_file = join(abspath(__file__), "../../../../configs/AVA/CTP/das6_32x2_112x112_R18_v2.2.yaml")
    cfg = load_config(args)

    # load model
    model = CTP(cfg)

    # test with sample inputs
    x = torch.randn(1, 3, 32, 112, 112)
    y_enc = model.encoder(x)
    assert y_enc.shape == (1, 512, 4, 7, 7)

    # 5 boxes for the 1st sample
    boxes = torch.randn(5, 4)
    boxes = torch.hstack([torch.zeros(5).view((-1, 1)), boxes])
    y = model([x], boxes)
    assert y.shape == torch.Size([5, 80])

    # check ckpt loading
    ckpt_path = cfg.MODEL.CKPT
    ckpt = torch.load(ckpt_path, map_location="cpu")

    esd = model.encoder.state_dict()
    csd = ckpt["state_dict"]

    layer_to_check = "layer4.0.conv2.conv_t.weight"
    X = esd[layer_to_check]
    Y = csd[f"backbone.{layer_to_check}"]
    assert (X == Y).all()
    print("Test passed!")
    
    print("::: Testing multi-label classification with Charades ...")
    # load cfg
    args.cfg_file = join(abspath(__file__), "../../../../configs/Charades/CTP/das6_32x8_112x112_R18.yaml")
    cfg = load_config(args)

    # load model
    model = CTP(cfg)

    # test with sample inputs
    x = torch.randn(1, 3, 32, 112, 112)
    y = model([x])
    assert y.shape == torch.Size([1, cfg.MODEL.NUM_CLASSES])
    print("Test passed!")



