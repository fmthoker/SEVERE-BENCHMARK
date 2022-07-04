import torch
import torch.nn as nn
import torchvision
import pdb



def diff(x):
    shift_x = torch.roll(x, 1, 2)
    return ((shift_x - x) + 1)/2


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


class PC_VideoNet(nn.Module):
    def __init__(self, base_network='r21d', feature_size=512, modality='rgb', class_seq=2, class_speed=4, head=True, proj_dim=128):
        """
        Args:
            feature_size (int): 512
        """
        super(PC_VideoNet, self).__init__()
        print(base_network)
        if base_network == 'r21d':
            model = torchvision.models.video.r2plus1d_18(pretrained=False, progress=True)
        elif base_network == 'r18':
            model = torchvision.models.video.r3d_18(pretrained=False, progress=True)
        elif base_network == 'c3d':
            model = C3D(with_classifier=False)
        elif base_network == 'r3d':
            model = R3D(layer_sizes=(1,1,1,1), with_classifier=False)
        elif base_network == 'r3d':
            model = I3D(with_classifier=False)
        
        if base_network == 'r21d' or base_network == 'r18':
            self.base_network = torch.nn.Sequential(*(list(model.children())[:-1]))
        else:
            self.base_network = model

        if modality == 'rgb':
            print('Use normal RGB clips for training')
            self.res = False
        else:
            print('[Warning]: use residual clips')
            self.res = True
        
        self.relu = nn.ReLU(inplace=True)

        self.head = head
        if self.head:
            self.projector = MoCoHead(512, proj_dim)

        self.fc_seq = nn.Linear(feature_size * 2, class_seq)
        self.fc_speed = nn.Linear(feature_size * 2, class_speed)

    def forward(self, x1, x2):
        if not self.res:
            f1 = self.base_network(x1)
            f2 = self.base_network(x2)
        else:
            f1 = self.base_network(diff(x1))
            f2 = self.base_network(diff(x2))

        f1 = f1.view(-1, 512)
        f2 = f2.view(-1, 512)

        h = torch.cat((f1, f2), dim=1)
        # sequence order
        y1 = self.fc_seq(h)
        # sequence speed, view1 is anchor
        y2 = self.fc_speed(h)
        # using projection head, contrastive learning
        if self.head:
            f1 = self.projector(f1)
            f2 = self.projector(f2)
        return f1, f2, y1, y2


class R18(nn.Module):
    def __init__(self, with_classifier=False, num_classes=101):
        super(R18, self).__init__()
        model = torchvision.models.video.r3d_18(pretrained=False, progress=True)
        self.base_network = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.with_classifier = with_classifier
        if with_classifier:
            self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_network(x)
        x = x.view(-1, 512)
        if self.with_classifier:
            x = self.linear(x)
        return x


class R21D(nn.Module):
    def __init__(self, with_classifier=False, num_classes=101):
        super(R21D, self).__init__()
        model = torchvision.models.video.r2plus1d_18(pretrained=False, progress=True)
        self.base_network = torch.nn.Sequential(*(list(model.children())[:-2]))
        self.with_classifier = with_classifier
        if with_classifier:
            self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_network(x)
        # x = x.view(-1, 512)
        if self.with_classifier:
            x = self.linear(x)
        return x

    def get_feature(self, x):
        #print("input size",x.size())
        x = self.base_network(x)
        #print("feature size",x.size())
        return x

class I3D(nn.Module):
    def __init__(self, with_classifier=False, num_classes=101):
        super(I3D, self).__init__()
        self.base_network = i3d()
        self.with_classifier = with_classifier
        if with_classifier:
            self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.base_network(x)
        x = x.view(-1, 1024)
        if self.with_classifier:
            x = self.linear(x)
        return x
