import torch
from torchvision import transforms
from torchvision import datasets
import numpy as np
import  random
import os
from PIL import Image
import shutil


# Load MNIST
def load_MNIST():
    # MNIST Download
    # data preprocess:
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0.1307,std=0.3081)]
    )
    # get Dataset
    test_dataset = datasets.MNIST(root=r'D:\\InternWorkspace_wsr\\dataset',train=False,download=True,transform=transform)
    return test_dataset


# Generate Multiple Digit image:
def Generate_MuiltipleDigit(dataset):
    num = dataset.data.shape[0]
    ori_height = dataset.data.shape[1] # 28
    ori_width = dataset.data.shape[2] # 28
    # generate image with fixed size
    length = 11 # phone number
    X_gen = np.zeros((num,ori_height,length*ori_width),dtype='uint8')
    Y_gen = np.zeros((num,length),dtype='int64')

    # generate image
    for i in range(num):
        for j in range(length):
            # select number
            idx = random.randint(0,num-1)
            X_gen[i,:,j*ori_width:(j+1)*ori_width] = dataset.data[idx]
            Y_gen[i,j] = dataset.targets[idx]
    
    return X_gen,Y_gen

# save as Image:
def save_as_image(X,Y,path,path_img):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path_img):
        os.mkdir(path_img)
    else:
        # 先清空里面的文件
        shutil.rmtree(path_img)
        os.mkdir(path_img)
    # save img
    for i in range(len(X)):
        img = Image.fromarray(X[i])
        img.save(os.path.join(path_img,str(i)+'.png'))

    # save target
    np.save(os.path.join(path,'label'),Y)

    print("Save Dataset Successfully!")


def generate(args):
    # load MNIST
    MNIST_test_dataset = load_MNIST()
    # generate test img:
    X_test,Y_test = Generate_MuiltipleDigit(MNIST_test_dataset)
    # save test img:
    save_as_image(X_test,Y_test,args.data_path,args.data_path+'/img')
