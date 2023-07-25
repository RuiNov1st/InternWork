from torchvision import transforms
import numpy as np
import os
import cv2
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader


# load test image and target:
def Read_MultipleDigit(path):
    filelist = os.listdir(path+'/img')
    img = cv2.imread(os.path.join(path+'/img',str(0)+'.png'))
    target = np.load(os.path.join(path,'label.npy'))

    data = np.zeros((len(filelist),img.shape[0],img.shape[1],img.shape[2]),dtype='uint8')
   
    for i in range(len(filelist)):
        data[i] = cv2.imread(os.path.join(path+'/img',str(i)+'.png'))
    return data,target

# transform image:
def get_transform():
    transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Lambda(lambda x:x.repeat(3,1,1)),
       transforms.Resize(224)])
    return transform


# seperate every single digit:
def Segment(img,target):
    Borderlist = []
   
    # 识别轮廓
    # cv2.RETR_EXTERNAL只检测外轮廓  cv2.CHAIN_APPROX_NONE存储所有的轮廓点
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 检索外部轮廓

    contours = list(contours)
    # 检查轮廓数量
    # 若分割数量少于应分割数量，则认为分割失败
    if len(contours)<len(target):
        print("Can not Split Correctly!")
        return []
    #  若分割数量大于应分割数量，则认为分割了噪点，假设噪点小于数字轮廓，则按照轮廓大小排序去除噪点。
    elif len(contours)>len(target):
        areas = [cv2.contourArea(cnt) for cnt in contours]
        contours_tuple = [(areas[i],contours[i]) for i in range(len(contours))]
        contours_tuple = sorted(contours_tuple,key=lambda x:x[0],reverse=True)
        contours = [cnt[1] for cnt in contours_tuple[:len(target)]]

    # 加入边框
    for cnt in contours:
        # 计算矩形
        peri = cv2.arcLength(cnt, True)  # 计算周长
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 计算有多少个拐角
        x, y, w, h = cv2.boundingRect(approx)  # 得到外接矩形的大小，返回的是左上角坐标和矩形的宽高
        # 最小边框框出来的图形：
        imgGet = img[y:y + h, x:x + w]
        
        # resize：
        # 若图片较大，则不应resize为MNIST数据集大小，否则失真。
        if imgGet.shape[0]*imgGet.shape[1] > 22500:
            imgGet = cv2.resize(imgGet,(150,150))
            size = 224
            h,w = 150,150
        else: # 按照MNIST数据集尺寸Resize
            size = 28
            
        # 需要往外扩边框
        top = max(0,(size-h)//2)
        bottom = max(0,size-top-h)
        left = max(0,(size-w)//2)
        right = max(0,size-w-left)

        # 补齐为正方形
        # 相应方向上的边框宽度：top bottom left right
        imgGet = cv2.copyMakeBorder(imgGet, top, bottom, left,right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        xx = x - left
        yy = y - top
        ss = size
        if imgGet.shape[0]!=28:
            imgGet = cv2.resize(imgGet,(28,28))
        Temptuple = (imgGet, xx, yy, ss)  # 将图像及其坐标放在一个元组里面，然后再放进一个列表里面就可以访问了
        Borderlist.append(Temptuple)

    # 对Borderlist按照数字顺序排序：
    Borderlist =  sorted(Borderlist,key=lambda x:(x[1],x[2]))
    
    return Borderlist


# data process:
# including segment and preprocess
def data_process(data,targets,handwrite=False):
    transform = get_transform()
    if handwrite:
        target_a = np.array([int(i) for i in targets])
        targets = torch.tensor(target_a)

        img = data
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
        Borderlist = Segment(img,targets)
        # transform:
        if len(Borderlist)!=0:
            Borderlist = [transform(x[0]) for x in Borderlist]
        
        return Borderlist,targets
    else:
        digit_data = []
        targets_n = []
        for i in range(data.shape[0]):
            img = data[i]
            # to gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # segment:
            Borderlist = Segment(img,targets[i])
            # transform:
            if len(Borderlist)!=0:
                Borderlist = [transform(x[0]) for x in Borderlist]
                digit_data.append(Borderlist)
                targets_n.append(targets[i]) # 防止轮廓分割错误
        targets = np.array(targets_n)
        targets = torch.tensor(targets)
        return digit_data,targets

    