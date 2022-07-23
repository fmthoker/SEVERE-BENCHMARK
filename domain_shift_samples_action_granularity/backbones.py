import re
import os
import torch.nn as nn

try:
    from termcolor import colored
except:
    os.system("pip install termcolor")
    from termcolor import colored

import torch
#from torchsummary import summary

import torchvision.models.video as video_models


def color(string: str, color_name: str = 'yellow') -> str:
    """Returns colored string for output to terminal"""
    return colored(string, color_name)


def print_update(message: str, width: int = 140, fillchar: str = ":") -> str:
    """Prints an update message
    Args:
        message (str): message
        width (int): width of new update message
        fillchar (str): character to be filled to L and R of message
    Returns:
        str: print-ready update message
    """
    message = message.center(len(message) + 2, " ")
    print(color(message.center(width, fillchar)))


def _check_inputs(backbone, init_method, ckpt_path):
    """Checks inputs for load_backbone()."""
    assert backbone in video_models.__dict__, f"{backbone} is not a valid backbone."
    assert init_method in [
        "scratch",
        "supervised",
        "ctp",
        "gdt",
        "rspnet",
        "tclr",
        "pretext_contrast",
        "video_moco",
        "moco",
        "selavi",
        "avid_cma",
    ]
    
    if init_method in ["scratch", ]:
        assert ckpt_path is None, f"{init_method} cannot be initialized from a checkpoint."\
            f"Pass ckpt_path=None while using init_method={init_method}."
    else:
        assert ckpt_path is not None, f"{init_method} must be initialized from a checkpoint."
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} does not exist.")


def load_supervised_checkpoint(ckpt_path, verbose=False):

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} does not exist.")

    csd = torch.load(ckpt_path, map_location=torch.device("cpu"))

    return csd


def load_ctp_checkpoint(ckpt_path, verbose=False):
    """
    Loads CTP checkpoint.
    Args:
        ckpt_path (str): path to checkpoint
        verbose (bool, optional): whether to print out the loaded checkpoint. Defaults to False.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} does not exist.")
    
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = checkpoint["state_dict"]
    
    # preprocess the state dict to make it compatible with R2+1D backbone
    csd = {k.replace("backbone.", ""):v for k,v in csd.items()}
    mapping = {
        "stem.conv_s": "stem.0",
        "stem.bn_s": "stem.1",
        "stem.conv_t": "stem.3",
        "stem.bn_t": "stem.4",
        "layer\\d{1}.\\d{1}.conv1.conv_s": "layer\\d{1}.\\d{1}.conv1.0.0",
        "layer\\d{1}.\\d{1}.conv1.bn_s": "layer\\d{1}.\\d{1}.conv1.0.1",
        "layer\\d{1}.\\d{1}.conv1.relu_s": "layer\\d{1}.\\d{1}.conv1.0.2",
        "layer\\d{1}.\\d{1}.conv1.conv_t": "layer\\d{1}.\\d{1}.conv1.0.3",
        "layer\\d{1}.\\d{1}.bn1": "layer\\d{1}.\\d{1}.conv1.1",
        "layer\\d{1}.\\d{1}.conv2.conv_s": "layer\\d{1}.\\d{1}.conv2.0.0",
        "layer\\d{1}.\\d{1}.conv2.bn_s": "layer\\d{1}.\\d{1}.conv2.0.1",
        "layer\\d{1}.\\d{1}.conv2.relu_s": "layer\\d{1}.\\d{1}.conv2.0.2",
        "layer\\d{1}.\\d{1}.conv2.conv_t": "layer\\d{1}.\\d{1}.conv2.0.3",
        "layer\\d{1}.\\d{1}.bn2": "layer\\d{1}.\\d{1}.conv2.1",
        "layer\\d{1}.\\d{1}.downsample": "layer\\d{1}.\\d{1}.downsample.0",
        "layer\\d{1}.\\d{1}.downsample_bn": "layer\\d{1}.\\d{1}.downsample.1",
        "layer\\d{1}.\\d{1}.downsample.conv": "layer\\d{1}.\\d{1}.downsample.0",
        "layer\\d{1}.\\d{1}.downsample_bn": "layer\\d{1}.\\d{1}.downsample.1",
    }
    
    # obtain mapping from checkpoint keys to backbone keys
    csd_keys_to_bsd_keys = dict()
    for k in csd.keys():
        
        for x in mapping:
            pattern = re.compile(x)
            if pattern.match(k):
                if x.startswith("layer"):
                    ori = ".".join((x.split(".")[2:]))
                    new = ".".join((mapping[x].split(".")[2:]))
                    replaced = k.replace(ori, new)
                else:
                    ori = x
                    new = mapping[x]
                    replaced = k.replace(ori, new)
                    
                disp = "\t\t".join([k, ori, new, replaced])
                if verbose:
                    print(disp)

                csd_keys_to_bsd_keys[k] = replaced

    # construct a new state dict that is fully compatible with R2+1D backbone
    new_csd = dict()
    for k,v in csd.items():
        if k in csd_keys_to_bsd_keys:
            new_csd[csd_keys_to_bsd_keys[k]] = csd[k]
        else:
            new_csd[k] = csd[k]

    return new_csd


def load_gdt_checkpoint(ckpt_path, verbose=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = ckpt["model"]
    
    # filter out audio network related keys
    csd = {k:v for k,v in csd.items() if not k.startswith(("audio_network", "mlp_a"))}
    
    # define mapping from csd keys to backbone keys
    mapping = lambda x: x.replace("video_network.base.", "")
    
    # construct a new state dict that is fully compatible with R2+1D backbone
    new_csd = {mapping(k):v for k,v in csd.items()}
    
    return new_csd


def load_rspnet_checkpoint(ckpt_path, verbose=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = ckpt["model"]
    
    # filter out encoder_k related keys    
    prefix = 'encoder_q.'
    blacklist = ['fc.', 'linear', 'head', 'new_fc', 'fc8', 'encoder_fuse']

    def filter(k):
        return k.startswith(prefix) and not any(k.startswith(f'{prefix}{fc}') for fc in blacklist)

    new_csd = {k[len(prefix):]: v for k, v in csd.items() if filter(k)}
    
    # remove prefixes
    new_csd = {k.replace("encoder.base_network.", "layer"): v for k, v in new_csd.items()}
    # replace layer0 with stem
    new_csd = {k.replace("layer0.", "stem."): v for k, v in new_csd.items()}
    
    return new_csd


def load_tclr_checkpoint(ckpt_path, verbose=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = ckpt["state_dict"]
    
    #bsd = video_models.r2plus1d_18().state_dict()
    
    # replace stem keys
    csd = {k.replace("module.0.stem", "stem"):v for k,v in csd.items()}

    for prefix in  ["layer1","layer2","layer3","layer4"]:

        csd = { k.replace(f"module.0.{prefix}", f"{prefix}"):v for k,v in csd.items()}
        #csd = { k.replace(f"module.0.{prefix}", f"{prefix}") \
              #for k in csd.keys() if k.startswith(f"module.0.{prefix}") }
    return csd
    
    # replace layer1 keys
    #prefix = "layer1"
    #csd_subset = [
    #    k.replace(f"module.0.{prefix}", f"{prefix}") \
    #    for k in csd.keys() if k.startswith(f"module.0.{prefix}")
    #]
    #bsd_subset = [x for x in bsd.keys() if x.startswith(prefix)]
    #assert set(csd_subset) == set(bsd_subset)

    #prefix = "layer2"
    #csd_subset = [
    #    k.replace(f"module.0.{prefix}", f"{prefix}") \
    #    for k in csd.keys() if k.startswith(f"module.0.{prefix}")
    #]
    #bsd_subset = [x for x in bsd.keys() if x.startswith(prefix)]
    #assert set(csd_subset) == set(bsd_subset)

    #prefix = "layer3"
    #csd_subset = [
    #    k.replace(f"module.0.{prefix}", f"{prefix}") \
    #    for k in csd.keys() if k.startswith(f"module.0.{prefix}")
    #]
    #bsd_subset = [x for x in bsd.keys() if x.startswith(prefix)]
    #assert set(csd_subset) == set(bsd_subset)

    #prefix = "layer4"
    #csd_subset = [
    #    k.replace(f"module.0.{prefix}", f"{prefix}") \
    #    for k in csd.keys() if k.startswith(f"module.0.{prefix}")
    #]
    #bsd_subset = [x for x in bsd.keys() if x.startswith(prefix)]
    #assert set(csd_subset) == set(bsd_subset)
    #
    ## TODO: There is a discrepancy between the number of layers in the csd and the bsd.
    ## This is because the TCLR checkpoint has few layers less than the R2+1D backbone.
    ## This shall be fixed in the future.
    #raise NotImplementedError


def load_pretext_contrast_checkpoint(ckpt_path, verbose=False):
    csd = torch.load(ckpt_path, map_location=torch.device("cpu"))
    
    # define key mapping
    csd = {k.replace("module.base_network.0", "stem"): v for k, v in csd.items()}
    csd = {k.replace("module.base_network.", "layer"):v for k,v in csd.items()}
    
    return csd


def load_video_moco_checkpoint(ckpt_path, verbose=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = ckpt["state_dict"]
    
    # filter out encoder_k keys and remove module.encoder_q.
    csd = {k.replace("module.encoder_q.", ""): v for k, v in csd.items() if not k.startswith("module.encoder_k.")}
    
    # remove fc keys
    csd = {k:v for k,v in csd.items() if not k.startswith("fc.")}
    
    return csd


def load_moco_checkpoint(ckpt_path, verbose=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = ckpt["state_dict"]
    
    # filter out encoder_k keys and remove module.encoder_q.
    csd = {k.replace("module.encoder_q.", ""): v for k, v in csd.items() if not k.startswith("module.encoder_k.")}
    
    # remove fc keys
    csd = {k:v for k,v in csd.items() if not k.startswith("fc.")}
    
    return csd


def load_selavi_checkpoint(ckpt_path, verbose=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = ckpt["model"]
    
    # filter only video base network keys
    csd = {
        k.replace("module.video_network.base.", ""):v for k,v in \
        csd.items() if k.startswith("module.video_network.base.")
    }
    
    return csd


def load_avid_cma_checkpoint(ckpt_path, verbose=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = ckpt["model"]
    
    # filter only video based network keys
    csd = {
        k.replace("module.video_model.", ""):v for k,v in \
        csd.items() if k.startswith("module.video_model.")
    }
    
    return csd


def load_backbone(backbone="r2plus1d_18", init_method="scratch", ckpt_path=None):
    """
    Loads given backbone (e.g. R2+1D from `torchvision.models`) with weights
    initialized from given VSSL method checkpoint.
    Args:
        backbone (str, optional): Backbone from `torchvision.models`.
            Defaults to "r2plus1d_18".
        init_method (str, optional): VSSL methods from which to initialize weights.
            Defaults to "scratch".
        ckpt_path ([str, None], optional): path to checkpoint for the given VSSL method.
            Defaults to None.
    """

    _check_inputs(backbone, init_method, ckpt_path)
    
    #backbone = getattr(video_models, backbone)(pretrained=(init_method == "supervised"))
    backbone = getattr(video_models, backbone)(pretrained=False)

    message = f"Checkpoint path not needed for {init_method} backbone."

    if init_method == 'tclr':
           backbone.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                       stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
           backbone.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                        kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)
    
    if init_method not in ["scratch", ] :
        state_dict = eval(f"load_{init_method.lower()}_checkpoint")(ckpt_path)
        print_update(f"Loading {init_method} checkpoint")
        message = backbone.load_state_dict(state_dict, strict=False)
        print("Checkpoint Path: {}".format(ckpt_path))
        print("Checkpoint Message: {}".format(message))
    else:
        print_update(f"Training from scratch")

    return backbone


if __name__ == "__main__":
    
    # setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_backbone("r2plus1d_18", "scratch")

    # Print summary
    # summary(model.to(device), (3, 16, 112, 112))

    # test AVID-CMA
    model = load_backbone(
        "r2plus1d_18",
        "AVID_CMA",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/avid_cma/avid_cma_ckpt-ep20.pth.tar",
    )

    # test SeLaVi
    model = load_backbone(
        "r2plus1d_18",
        "SeLaVi",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/selavi/selavi_kinetics.pth",
    )

    # test MoCo
    model = load_backbone(
        "r2plus1d_18",
        "MoCo",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/moco/checkpoint_0199.pth.tar",
    )

    # test PretextContrast
    model = load_backbone(
        "r2plus1d_18",
        "VideoMoCo",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/video_moco/r2plus1D_checkpoint_0199.pth.tar",
    )

    # test PretextContrast
    model = load_backbone(
        "r2plus1d_18",
        "PretextContrast",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/pretext_checkpoint/pcl_r2p1d_res_ssl.pt",
    )

    # test TCLR
    model = load_backbone(
         "r2plus1d_18",
         "TCLR",
         ckpt_path="/home/pbagad/models/checkpoints_pretraining/tclr/rpd18kin400.pth",
     )

    # test RSPNet
    model = load_backbone(
        "r2plus1d_18",
        "RSPNet",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar",
    )
    
    # test GDT
    model = load_backbone(
        "r2plus1d_18",
        "GDT",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/gdt/gdt_K400.pth",
    )
    
    # test CTP
    model = load_backbone(
        "r2plus1d_18",
        "CTP",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth",
    )
