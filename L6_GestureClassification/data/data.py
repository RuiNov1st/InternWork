import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os


output_labels = ['like','dislike']

def data_dataload(output_labels,data_path,name):
    array = []
    labels = []
    for label in output_labels:
        data =  np.load(os.path.join(data_path,label,'{}_{}.npz'.format(label,name)))
        data_X = data['x']
        data_y = data['y']
        array.extend(data_X)
        labels.extend(data_y)
    array = np.array(array)
    labels = np.array(labels)
    return array,labels

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
    [transforms.ToTensor()]
    )

    return transform



def load_data(args):
    data_path = args.data_path
    # load dataset: (224,24,3) like and dislike
    train_data,train_label =data_dataload(output_labels,data_path,'train')
    val_data,val_label = data_dataload(output_labels,data_path,'valid')
    test_data,test_label= data_dataload(output_labels,data_path,'test')
    # dataset and dataloader:
    transform = get_transform() # just to tensor
    train_dataset,val_dataset,test_dataset = Dataset(train_data,train_label,transform),Dataset(val_data,val_label,transform),Dataset(test_data,test_label,transform)
    # dataloader: 
    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size = args.batch_size,
                                    shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size = args.batch_size,shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size = 1,
                                    shuffle=True)
    
    return train_dataloader,val_dataloader,test_dataloader

    

