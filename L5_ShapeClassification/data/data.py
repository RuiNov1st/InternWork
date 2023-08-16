from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os

# 数据集定义
output_labels = [
  'circle',   #    0
  'rectangle', #    1
  'triangle', # 2
  'star']  # 3


# 载入数据集
def load_data(path):
    train_data= np.load(os.path.join(path,'train.npz'))
    train_X,train_y = train_data['x'],train_data['y']
    val_data= np.load(os.path.join(path,'val.npz'))
    val_X,val_y = val_data['x'],val_data['y']
    test_data= np.load(os.path.join(path,'test.npz'))
    test_X,test_y = test_data['x'],test_data['y']

    print("train_X shape: ", train_X.shape)
    print("train_y shape: ", train_y.shape)
    print("val_X shape: ", val_X.shape)
    print("val_y shape: ", val_y.shape)
    print("test_X shape: ", test_X.shape)
    print("test_y shape: ", test_y.shape)

    return train_X,train_y,val_X,val_y,test_X,test_y


# 封装数据集
# ReWrite Dataset
class Dataset(Dataset):
    def __init__(self,data,target,transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        if self.transform:
            data = self.transform(data)
        return data,target
    
    def __len__(self):
        return len(self.data)


# transform image:
def get_transform():
    transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Lambda(lambda x:x.repeat(3,1,1)),
        # 加一些数据增强
        transforms.RandomRotation(90), # 随机旋转
        transforms.RandomHorizontalFlip(p=0.5), # 水平翻转
        transforms.RandomVerticalFlip(p=0.5), # 竖直翻转
       transforms.Resize(224,antialias=True)])
    return transform


# 返回Dataloader
def data_process(args):
    # load dataset:
    train_X,train_y,val_X,val_y,test_X,test_y = load_data(args.data_path)
    transform = get_transform()
    train_dataset,val_dataset,test_dataset = Dataset(train_X,train_y,transform),Dataset(val_X,val_y,transform),Dataset(test_X,test_y,transform)
    # dataloader:
    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size = args.batch_size,
                                    shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size = args.batch_size,shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size = 1,
                                    shuffle=True)

    return train_dataloader,val_dataloader,test_dataloader