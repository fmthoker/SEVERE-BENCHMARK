import numpy as np 
import torch.nn as nn
import torch
import torchvision
#from torchsummary import summary

from torchvision.models.utils import load_state_dict_from_url
from networks.tclr.r21d_prp_modified import R2Plus1DNet, mlp


def build_r2plus1d_prp_backbone():
    model = R2Plus1DNet(layer_sizes= (1,1,1,1), with_classifier=False, return_conv=False)

    model.conv5.block1.conv2.spatial_conv = nn.Conv3d(512, 1152, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0), bias=False)
    model.conv5.block1.downsampleconv.spatial_conv = nn.Conv3d(256, 170, kernel_size=(1, 1, 1), stride=(1, 6, 6), bias=False) #I AM NOT SURE ABOUT THE BIG 6X6 STRIDE

    model.conv5.block1.downsampleconv.temporal_conv = nn.Conv3d(170, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    model.conv5.block1.conv1.temporal_conv = nn.Conv3d(921, 512, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0), dilation = (2, 1, 1), bias=False)

    return model


def build_r2plus1d_prp_mlp(num_classes):
    f = build_r2plus1d_prp_backbone()
    linear = nn.Linear(512, num_classes)
    model = nn.Sequential(f,linear)
    return model


def load_r2plus1d_prp_mlp(saved_model_file,num_classes):
    model = build_r2plus1d_prp_mlp(num_classes)
    pretrained = torch.load(saved_model_file)
    pretrained_kvpair = pretrained['state_dict']
    model_kvpair = model.state_dict()

    #print("model keys",list(model.state_dict().keys()))
    #print("dict keys",list(model_kvpair.keys()))
    for layer_name, weights in pretrained_kvpair.items():
        # if 'module.1' in layer_name: # removing embedder part which is module.1 in the model+embedder
        #     continue
        layer_name = layer_name.replace('module.','')
        model_kvpair[layer_name] = weights  

    msg = model.load_state_dict(model_kvpair, strict=False)
    print(msg)
    #print(f'{saved_model_file} loaded successfully')
    
    return model 


if __name__ == '__main__':
    from torch.cuda.amp import autocast

    # model = C3D(num_classes=102)    
    # model = nn.Sequential(model,mlp())
    model = build_r2plus1d_prp_mlp()
    # print(model)
    model = model.cuda()
    input = torch.rand(8, 3, 16, 112, 112).cuda()
    # output = model(input)
    with autocast():

        output = model((input,'s'))

    print(len(output))
    print(output[0].shape)

    # summary(model, (3,16, 112,112))
