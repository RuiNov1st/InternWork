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

# 噪点合并与拼接
# 假设手写数字的高度相同，我们认为小于1/3平均高度的轮廓是噪点

# 找到距离噪点最近的非噪点图像，用于将噪点合并
def find_zero(arr_list,idx,left=True):
    if left:
        for i in range(idx,-1,-1):
            if arr_list[i]==0:
                return i
        return -1
    else:
        for i in range(idx,len(arr_list),1):
            if arr_list[i]==0:
                return i
        return -1

# 筛选噪点并进行合并
def remove_noise(cnt_merge):
    # 设置噪点的高度阈值为图像平均高度的1/3
    height_mean = np.mean([cnt[4] for cnt in cnt_merge])
    threshold = 1/3*height_mean
    
    merge_idx = [0 for i in range(len(cnt_merge))] # 噪点标记，0为非噪点，1为噪点
    merge_idxx = [-1 for i in range(len(cnt_merge))] # 噪点标记合并的位置

    # 先标记噪点
    for cnt in range(len(cnt_merge)):
        # 小于2/3平均高度的值是噪点
        if cnt_merge[cnt][4]<threshold:
            merge_idx[cnt] = 1 # 噪点
    
    # 确定噪点合并到哪边
    for cnt in range(len(cnt_merge)):
        # 一个点是噪点，则需要找到左边离它最近的不是噪点的数和右边离它最近的不是噪点的数
        if merge_idx[cnt]==1:
            # 噪音在最左边
            if cnt==0:
                right_idx = find_zero(merge_idx,cnt+1,False)
                if right_idx == -1:
                    print("Cant find valid digit!")
                else:
                    merge_idxx[cnt] = right_idx
            # 噪音在最右边
            elif cnt == len(cnt_merge)-1:
                left_idx = find_zero(merge_idx,cnt-1,True)
                if left_idx == -1:
                    print("Cant find valid digit!")
                else:
                    merge_idxx[cnt] = left_idx
            # 噪音在中间
            else:
                left_idx = find_zero(merge_idx,cnt-1,True)
                right_idx = find_zero(merge_idx,cnt+1,False)
                if left_idx == -1 and right_idx==-1:
                    print("Cant find valid digit!")
                elif left_idx==-1:
                    merge_idxx[cnt] = right_idx
                elif right_idx==-1:
                    merge_idxx[cnt] = left_idx
                else:
                    # 看离哪个离得近：
                    left_width = cnt_merge[cnt][1]-cnt_merge[left_idx][1]-cnt_merge[left_idx][3]
                    right_width = cnt_merge[right_idx][1]-cnt_merge[cnt][1]-cnt_merge[cnt][3]
                    if left_width<right_width:
                        merge_idxx[cnt] = left_idx
                    else:
                        merge_idxx[cnt] = right_idx
    
    # 噪点合并：
    cnt_rm = cnt_merge.copy()
    for i in range(len(merge_idxx)):
        # 不合并，包括没有找到合并点的
        if merge_idxx[i]==-1:
            continue
        else:
            # 轮廓拼接
            cnt_rm[merge_idxx[i]][0] = np.vstack((cnt_merge[merge_idxx[i]][0],cnt_merge[i][0]))
            # x
            temp_x = cnt_rm[merge_idxx[i]][1]
            cnt_rm[merge_idxx[i]][1] = min(cnt_merge[merge_idxx[i]][1],cnt_merge[i][1])
            # y
            temp_y = cnt_rm[merge_idxx[i]][2]
            cnt_rm[merge_idxx[i]][2] = min(cnt_merge[merge_idxx[i]][2],cnt_merge[i][2])
            # w
            cnt_rm[merge_idxx[i]][3] = max(temp_x+cnt_rm[merge_idxx[i]][3],cnt_merge[i][1]+cnt_merge[i][3])-cnt_rm[merge_idxx[i]][1]
            # h
            cnt_rm[merge_idxx[i]][4] = max(temp_y+cnt_rm[merge_idxx[i]][4],cnt_merge[i][2]+cnt_merge[i][4])-cnt_rm[merge_idxx[i]][2]
            # s
            cnt_rm[merge_idxx[i]][5] = cnt_rm[merge_idxx[i]][3]*cnt_rm[merge_idxx[i]][4]
            
            # 合并之后，把噪点位置清空
            cnt_rm[i] = None

    # 然后把None的去掉
    cnt_rmm = []
    for i in cnt_rm:
        if i is None:
            continue
        else:
            cnt_rmm.append(i)
        
    return cnt_rmm


# 形态学方法：
# 开运算：先腐蚀后膨胀
def Open_operation(img,kernel):
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return open

# 闭运算：先膨胀后腐蚀
def Closed_operation(img,kernel):
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed

# 腐蚀运算
def Erode_operation(img,kernel,iter=1):
    eroded = cv2.erode(img,kernel,iterations=iter)
    return eroded

# 膨胀运算
def Dilate_operation(img,kernel,iter=1):
    dilate = cv2.dilate(img,kernel,iterations=iter)
    return dilate

# 形态学方法去除噪点：
# 在这里使用2*2的腐蚀操作和1*1的膨胀操作各迭代1次对图像进行处理：
def morph(img):
    # 先腐蚀后膨胀
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    data = Erode_operation(img,kernel1,1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    data = Dilate_operation(data,kernel2,1)
    return data



# seperate every single digit: fixed length
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

    transform = get_transform()
    Borderlist = [transform(x[0]) for x in Borderlist]
            
    
    return Borderlist

# seperate every single digit: variable length
def Segment_variable(img):
    # 边框提取和噪点合并：
    imgg = img.copy()
    # 形态学方法+高斯滤波去除噪点
    imgg = morph(imgg)
    imgg = cv2.GaussianBlur(imgg,ksize=(5,5),sigmaX=0)
    # 数字边框提取：
    # cv2.RETR_EXTERNAL只检测外轮廓  cv2.CHAIN_APPROX_NONE存储所有的轮廓点
    contours, hierarchy = cv2.findContours(imgg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 检索外部轮廓

    # 计算外接矩形：
    cnt_data = Rectangle_compute(contours)
    # 若外接矩形之间存在交集，则需要进行轮廓合并：
    # 对轮廓按外接矩形进行排序
    cnt_sort = sorted(cnt_data,key=lambda z:(z[1],z[2]),reverse=False)
    # 合并轮廓：若在x轴上有交集，则认为两个轮廓可以合并
    cnt_merge = merge_Contours(cnt_sort)
    # 噪点合并：将高度没有达到阈值的小噪点合并至其周围的正常数字中。
    cnt_merge = remove_noise(cnt_merge)
   
    # 数字居中及边框添加处理：
    cnt_data = cnt_merge
    Borderlist = []
    imgGet_list = []
    for cnt in cnt_data:
        x,y,w,h,s = cnt[1],cnt[2],cnt[3],cnt[4],cnt[5]
        # 最小边框框出来的图形：
        imgGet = img[y:y + h, x:x + w]
        imgGet_list.append(imgGet) # 未处理之前先保存
    
        size=28
        imgGet = MNIST_process(imgGet,h,w,origin_size=20,new_size=28)
        xx,yy,ss = x,y,size
        Temptuple = (imgGet, xx, yy, ss)  # 将图像及其坐标放在一个元组里面，然后再放进一个列表里面就可以访问了
        Borderlist.append(Temptuple)
        
    # 对Borderlist按照数字顺序排序：
    Borderlist =  sorted(Borderlist,key=lambda x:(x[1]))
    transform = get_transform()
    Borderlist = [transform(x[0]) for x in Borderlist]
    
    return Borderlist,imgGet_list


# data process:
# including segment and preprocess
def data_process(data,targets,handwrite=False,fixed_length=True):
    # 手写图片：
    if handwrite:
        target_a = np.array([int(i) for i in targets])
        targets = torch.tensor(target_a)

        img = data
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)

        # 固定长度识别
        if fixed_length:
            Borderlist = Segment(img,targets)
            return Borderlist,targets
        # 可变长度：
        else:
            Borderlist,imgGet_list = Segment_variable(img)
            return Borderlist,targets,imgGet_list
    else:
        digit_data = []
        targets_n = []
        miniimg_list = []
        for i in range(data.shape[0]):
            img = data[i]
            # to gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # segment:
            if fixed_length:
                Borderlist = Segment(img,targets[i])
            else:
                Borderlist,imgGet_list = Segment_variable(img)
                miniimg_list.append(imgGet_list)

            digit_data.append(Borderlist)
        if fixed_length:
            targets = torch.tensor(targets)
            return digit_data,targets
        else:
            return digit_data,targets,miniimg_list
        
        
    # # 固定长度识别
    # if fixed_length:
    #     if handwrite:
    #         target_a = np.array([int(i) for i in targets])
    #         targets = torch.tensor(target_a)

    #         img = data
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         _,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
    #         Borderlist = Segment(img,targets)
    #         # transform:
    #         if len(Borderlist)!=0:
    #             Borderlist = [transform(x[0]) for x in Borderlist]
            
    #         return Borderlist,targets
    #     else:
    #         digit_data = []
    #         targets_n = []
    #         for i in range(data.shape[0]):
    #             img = data[i]
    #             # to gray:
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #             # segment:
    #             Borderlist = Segment(img,targets[i])
    #             # transform:
    #             if len(Borderlist)!=0:
    #                 Borderlist = [transform(x[0]) for x in Borderlist]
    #                 digit_data.append(Borderlist)
    #                 targets_n.append(targets[i]) # 防止轮廓分割错误
    #         targets = np.array(targets_n)
    #         targets = torch.tensor(targets)
    #         return digit_data,targets
    # else:
    #     # 可变长度识别
    #     if handwrite:
    #         digit_target = np.array([int(i) for i in targets])
            
    #         img = data
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         _,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
    #         Borderlist,imgGet_list = Segment_variable(img)
            
    #         # transform:
    #         Borderlist = [transform(x[0]) for x in Borderlist]
            
    #         return Borderlist,digit_target,imgGet_list
    #     else:
    #         digit_data = []
    #         digit_target = []
    #         miniimg_list = []
    #         for i in range(data.shape[0]):
    #             img = data[i]
    #             # to gray:
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #             # segment:
    #             Borderlist,imgGet_list = Segment_variable(img)
    #             Borderlist = [transform(x[0]) for x in Borderlist]

    #             digit_data.append(Borderlist)
    #             digit_target.append(targets[i])
    #             miniimg_list.append(imgGet_list)

    #         return digit_data,digit_target,miniimg_list
    