import torch
from torch import nn
import os
import yaml

class Head(nn.Module):

    def __init__(self, fc_emd,num_classes):
 

            super(Head, self).__init__()

            self.fc_emd = fc_emd
            self.num_classes= num_classes

            self.fc_cls_1 = nn.Linear(fc_emd, 2*num_classes)
            self.fc_box_1 = nn.Linear(fc_emd, num_classes)
            self.fc_cls_2 = nn.Linear(fc_emd, 2*num_classes)
            self.fc_box_2 = nn.Linear(fc_emd, num_classes)

    def forward(self, x):
            new_x = x.reshape(-1, self.fc_emd)
            pred_cls_1 = self.fc_cls_1(new_x)
            pred_cls_1 = pred_cls_1.reshape(-1, 2, self.num_classes)
            pred_box_1 = self.fc_box_1(new_x)

            pred_cls_2 = self.fc_cls_2(new_x)
            pred_cls_2 = pred_cls_2.reshape(-1, 2, self.num_classes)
            pred_box_2 = self.fc_box_2(new_x)
            return pred_cls_1, pred_box_1, pred_cls_2, pred_box_2

def generate_model(opt):

    from backbones import load_backbone
    
    model = load_backbone("r2plus1d_18",opt.pretext_model_name,opt.pretext_model_path)
    model.fc = Head(fc_emd=512, num_classes = opt.n_classes)
    

    if not opt.no_cuda:
        #model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    return model, model.parameters()
