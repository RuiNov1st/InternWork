from torchvision import transforms
import numpy as np
import os
import cv2
import torch


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


# 基于MNIST数据集生成过程处理测试数据集
def MNIST_process(img,h,w,origin_size=20,new_size=28):
    # size normalized while preserving aspect ratio
    factor = origin_size/max(h,w)
    img_r = cv2.resize(img,(0,0),fx=factor,fy=factor)
    img = np.zeros((origin_size,origin_size),dtype='uint8')
    if img_r.shape[0]==origin_size:
        offset = (origin_size-img_r.shape[1])//2
        img[:,offset:offset+img_r.shape[1]] = img_r
    elif img_r.shape[1]==origin_size:
        offset = (origin_size-img_r.shape[0])//2
        img[offset:offset+img_r.shape[0],:] = img_r
    else:
        y_offset = (origin_size-img_r.shape[0])//2
        x_offset = (origin_size-img_r.shape[1])//2

        img[y_offset:y_offset+img_r.shape[0],x_offset:x_offset+img_r.shape[1]] = img_r

    # center of mass:
    sum_intensity = 0
    for x in range(origin_size):
        for y in range(origin_size):
            sum_intensity+=img[y,x]
    sum_x = 0
    for x in range(origin_size):
        for y in range(origin_size):
            sum_x+=x*img[y,x]
    sum_y = 0
    for y in range(origin_size):
        for x in range(origin_size):
            sum_y+=y*img[y,x]
    
    X_MassOfCenter = int(sum_x/sum_intensity)
    Y_MassOfCenter = int(sum_y/sum_intensity)
    
    # center:
    Center = origin_size//2
    X_offset = Center - X_MassOfCenter
    Y_offset = Center - Y_MassOfCenter
    # new_img:
    new_img = np.zeros((new_size,new_size),dtype='uint8')
    start = (new_size-origin_size)//2
    new_img[start:start+origin_size,start:start+origin_size] = img
    # shift:
    M = np.float32([[1,0,X_offset],[0,1,Y_offset]])
    new_img = cv2.warpAffine(new_img,M,(new_img.shape[1],new_img.shape[0]))
    
    return new_img


# 计算轮廓的外接矩形：
def Rectangle_compute(contours):
    cnt_rect = []
    for cnt in range(len(contours)):
        peri = cv2.arcLength(contours[cnt], True)  # 计算周长
        approx = cv2.approxPolyDP(contours[cnt], 0.02 * peri, True)  # 计算有多少个拐角
        x, y, w, h = cv2.boundingRect(approx)  # 得到外接矩形的大小，返回的是左上角坐标和矩形的宽高
        cnt_rect.append([contours[cnt],x, y, w, h,w*h]) # [轮廓，x,y,w,h,矩形面积]
    return cnt_rect


# 合并轮廓
def merge_Contours(cnt_sort):
    last_x,last_w = 0,0
    cnt_merge = []
    for i in range(len(cnt_sort)):
        if i == 0:
            last_x = cnt_sort[i][1]
            last_w = cnt_sort[i][3]
            cnt_merge.append(cnt_sort[i])
        else:
            # 在x轴上存在交集：
            if cnt_sort[i][1]<last_x+last_w:
                # 轮廓拼接
                cnt_merge[-1][0] = np.vstack((cnt_merge[-1][0],cnt_sort[i][0]))
                # y,w,h，面积改变，x不用变
                temp_h = cnt_merge[-1][2]
                cnt_merge[-1][2] = min(cnt_merge[-1][2],cnt_sort[i][2]) # y取最上面那个
                cnt_merge[-1][3] = max(cnt_merge[-1][1]+cnt_merge[-1][3],cnt_sort[i][1]+cnt_sort[i][3])-cnt_merge[-1][1] # w取最大
                cnt_merge[-1][4] = max(temp_h+cnt_merge[-1][4],cnt_sort[i][2]+cnt_sort[i][4])-cnt_merge[-1][2] # h取最大
                cnt_merge[-1][5] = cnt_merge[-1][3]*cnt_merge[-1][4] # 面积根据改变后的wh来计算
                last_x = cnt_merge[-1][1] # x
                last_w = cnt_merge[-1][3] # w
            # 不存在交集
            else:
                last_x = cnt_sort[i][1]
                last_w = cnt_sort[i][3]
                cnt_merge.append(cnt_sort[i])
    
    return cnt_merge

# seperate every single digit:
def Segment(img,target):
    length = len(target)
    Borderlist = []
    # cv2.RETR_EXTERNAL只检测外轮廓  cv2.CHAIN_APPROX_NONE存储所有的轮廓点
    imgg = img.copy()
    contours, hierarchy = cv2.findContours(imgg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 检索外部轮廓
    
    # 计算外接矩形：
    cnt_data = Rectangle_compute(contours)

    # 若外接矩形之间存在交集，则需要进行轮廓合并：
    # 对轮廓按外接矩形进行排序
    cnt_sort = sorted(cnt_data,key=lambda z:(z[1],z[2]),reverse=False)
    # 合并轮廓：若在x轴上有交集，则认为两个轮廓可以合并
    cnt_merge = merge_Contours(cnt_sort)

    # 合并轮廓之后轮廓少于给定数量
    if len(cnt_merge)<length:
        print("Wrong Segment!")
        return 
    # 合并轮廓之后轮廓多于给定数量，按面积排序去除多余的
    if len(cnt_merge)>length:
        cnt_merge = sorted(cnt_merge,key=lambda z:z[5],reverse=True)
        cnt_merge = cnt_merge[:length]

    cnt_data = cnt_merge
    
    for cnt in cnt_data:
        x,y,w,h = cnt[1],cnt[2],cnt[3],cnt[4]
        fix_size=28

        # 最小边框框出来的图形：
        imgGet = img[y:y + h, x:x + w]
        
        imgGet = MNIST_process(imgGet,h,w)
        
        Temptuple = (imgGet, x,y,fix_size)  # 将图像及其坐标放在一个元组里面，然后再放进一个列表里面就可以访问了
        Borderlist.append(Temptuple)
        
    # 对Borderlist按照数字顺序排序：
    Borderlist =  sorted(Borderlist,key=lambda x:(x[1]))
    
    
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

    