from kaggle import api
import os
from pathlib import Path
import tarfile
import zipfile
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np


output_labels = ['like','dislike']

# dataset extract:
def file_extract(fname, dest=None):
    "Extract `fname` to `dest` using `tarfile` or `zipfile`."
    if dest is None: dest = Path(fname).parent
    fname = str(fname)
    if fname.endswith('gz'):  tarfile.open(fname, 'r:gz').extractall(dest)
    elif fname.endswith('zip'): zipfile.ZipFile(fname).extractall(dest)
    else: raise Exception(f'Unrecognized archive: {fname}')

# dataset download and extract:
# dataset:https://www.kaggle.com/datasets/innominate817/hagrid-classification-512p
def download(kaggle_dataset,archive_dir):
    archive_path = Path(f'{archive_dir}/{dataset_name}.zip')
    dataset_path = os.path.join(archive_dir,dataset_name)
    
    if not archive_path.exists():
        api.dataset_download_cli(kaggle_dataset, path=archive_dir)
    file_extract(fname=archive_path, dest=archive_dir)

    # remove other labels to save memory
    for label in os.listdir(dataset_path):
        if not label in output_labels:
            shutil.rmtree(os.path.join(dataset_path,label))
        

# data preprocess function:
class preprocess:
    def get_label_path(self,label,data_path):
        label_path = []
        pathh = os.path.join(data_path,label)
        filelist = os.listdir(pathh)
        for file in filelist:
            label_path.append(os.path.join(pathh,file))
        return sorted(list(set(label_path)))
    

    def train_val_test_split(self,label,data_path,train_test_factor = 2/10,train_val_factor=2/8):
        # 得到该label下所有的图片文件路径
        label_path = self.get_label_path(label,data_path)
        # 按比例分配数据集：6:2:2
        x_train,x_test = train_test_split(label_path,test_size=train_test_factor,shuffle=True)
        x_train,x_val = train_test_split(x_train,test_size=train_val_factor,shuffle=True)
    
        return x_train,x_val,x_test
    
    def load_save_data(self,data,output_labels,label,save_path,name):
        array = []
        for image_file in data:
            arr = cv2.imread(image_file)
            arr = cv2.resize(arr,((224,224))) # resize to 224*224
            arr = cv2.cvtColor(arr,cv2.COLOR_BGR2RGB) # retain 3 channels
            array.append(arr)
        
        array = np.array(array)
        labels = np.array([output_labels.index(label)]*len(array))

        # 先存为npz文件
        np.savez(os.path.join(save_path,label,'{}_{}'.format(label,name)),x=array,y=labels)

    # 存为数据集
    def make_dataset(self,data_path,label,output_labels,save_path):
        # 先划分数据集
        x_train,x_val,x_test = self.train_val_test_split(label,data_path)
        
        # # 保存数据
        # # train:
        self.load_save_data(x_train,output_labels,label,save_path,'train')
        # # valid:
        self.load_save_data(x_val,output_labels,label,save_path,'valid')
        # # test:
        self.load_save_data(x_test,output_labels,label,save_path,'test')
        print("{} save successfully!".format(label))
    

# data split and save：
def save_dataset(img_path,data_path):
    Path(data_path).mkdir(parents=True, exist_ok=True)
    preprocessor = preprocess()
    for label in output_labels:
        if not os.path.exists(os.path.join(data_path,label)):
            os.mkdir(os.path.join(data_path,label))
        preprocessor.make_dataset(img_path,label,output_labels,data_path)


if __name__ == '__main__':
    dataset_name = 'hagrid-classification-512p' # dataset name
    kaggle_dataset = f'innominate817/{dataset_name}' # kaggle dataset repo:
    archive_dir = './dataset/' # local dir: zip
    # download:
    if not os.path.exists(os.path.join(archive_dir,dataset_name)):
        download(kaggle_dataset,archive_dir)
    # process: to train/valid/test.npz
    dataset_dir = os.path.join(archive_dir,'data')
    save_dataset(os.path.join(archive_dir,dataset_name),dataset_dir)

