"""VCOPN"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def _init_weights(module, init_linear='normal'):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MoCoHead(nn.Module):
    '''The non-linear neck in MoCO v2: fc-relu-fc
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 norm=True):
        super(MoCoHead, self).__init__()
        if not hid_channels:
            hid_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))
        self.norm = norm
        if not norm:
            print('[Warning] Do not normalize features after projection.')
        self.l2norm = Normalize(2)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        if self.norm:
            x = self.l2norm(x)
        return x


class PCLNet(nn.Module):
    def __init__(self, base_network, feature_size, tuple_len=3, modality='rgb', class_num=5, proj_dim=512, head=False):
        """
        Args:
            feature_size (int): 512
        """
        super(PCLNet, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = class_num
        self.fc = nn.Linear(feature_size*tuple_len, self.class_num)
        if modality == 'rgb':
            #print('Use normal RGB clips for training')
            self.res = False
        else:
            #print('[Warning]: use residual clips')
            self.res = True
        
        self.relu = nn.ReLU(inplace=True)
        self.head = head
        if self.head:
            self.projector = MoCoHead(feature_size, proj_dim)

    def diff(self, x):
        shift_x = torch.roll(x, 1, 2)
        return shift_x - x
        #return ((shift_x - x) + 1)/2

    def forward(self, tuple):
        if not self.res:
          f1 = self.base_network(tuple[:, 0, :, :, :, :])
          f2 = self.base_network(tuple[:, 1, :, :, :, :])
          f3 = self.base_network(tuple[:, 2, :, :, :, :])
        else:
          f1 = self.base_network(self.diff(tuple[:, 0, :, :, :, :]))
          f2 = self.base_network(self.diff(tuple[:, 1, :, :, :, :]))
          f3 = self.base_network(self.diff(tuple[:, 2, :, :, :, :]))

        f1 = f1.view(-1, self.feature_size)
        f2 = f2.view(-1, self.feature_size)
        f3 = f3.view(-1, self.feature_size)
        h = torch.cat((f1, f2, f3), dim=1)
        h = self.fc(h)  # logits
        if self.head:
            f1 = self.projector(f1)
            f3 = self.projector(f3)
        return h, f1, f3