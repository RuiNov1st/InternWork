"""
Description: 处理MNIST数据并由dataloader封装并载入
Author: weishirui
Time: 2023/7/20
"""
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


# download MNIST and preprocess:
def get_MNIST(args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x:x.repeat(3,1,1)), # ResNet18要求输入图形为3通道，需将单通道设置为3通道
         transforms.Resize(224)]
    )
    train_dataset = datasets.MNIST(root=args.data_path,train=True,download=True,transform=transform)
    test_dataset = datasets.MNIST(root=args.data_path,train=False,download=True,transform=transform)
    return train_dataset,test_dataset


# return dataloader:
def get_dataloader(train_dataset,test_dataset,args):
    # dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size = args.batch_size,
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size = 1,
                                shuffle=True)
    return train_dataloader,test_dataloader