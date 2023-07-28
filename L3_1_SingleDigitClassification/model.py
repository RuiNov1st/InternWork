import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    # load ResNet18 from Pytorch
    def __init__(self,weight_path=None):
        super(ResNet18,self).__init__()
        self.resnet18 = None
        if weight_path:
            self.resnet18 = models.resnet18()
            resnet18_weights = torch.load(weight_path)
            self.resnet18.load_state_dict(resnet18_weights)
        else:
            self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.transform = models.ResNet18_Weights.DEFAULT.transforms(antialias=True) # ResNet18默认数据处理方法

    # modify parameters for finetune:
    def finetune_modify(self):
        resnet18_fc_features = self.resnet18.fc.in_features
        self.resnet18.fc = torch.nn.modules.Linear(resnet18_fc_features,10)
    
        # freeze gredient
        for param in self.resnet18.parameters():
            param.requires_grad = False
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True
        

    
           




        

