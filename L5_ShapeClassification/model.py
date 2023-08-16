import torch
import torch.nn as nn
import torchvision.models as models
from data.data import output_labels

class ResNet18(nn.Module):
    # load ResNet18 from Pytorch
    def __init__(self):
        super(ResNet18,self).__init__()
        self.resnet18 = models.resnet18()
        resnet18_fc_features = self.resnet18.fc.in_features
        self.resnet18.fc = torch.nn.modules.Linear(resnet18_fc_features,len(output_labels))