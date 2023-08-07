import torch
import argparse
from tqdm import tqdm
import os
import cv2
from data.data import Read_MultipleDigit,data_process,Segment_variable
from data.generate import generate
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


# 二次检测流程
def second_test(model,probability,predict_list,miniimg_list,use_cuda):
    # 3sigma筛选出需要二次检测的图像：
    mean_p = np.mean(probability)
    std_p = np.std(probability)
    sigma3 = max(0,mean_p-3*std_p)
    test2nd_idx = []
    for m in range(len(probability)):
        if probability[m]<sigma3:
            test2nd_idx.append(m)

    # 需要二次检测：
    if len(test2nd_idx)!=0:
        for m in test2nd_idx:
            # 使用划分得到的最小图像，而不使用处理后的test图像进行二次处理
            test_temp = miniimg_list[m]
            Borderlist,_ = Segment_variable(test_temp)
            
            # 无法切分，则说明只有一个
            if len(Borderlist)==1:
                # 保留下来
                pass
            # 能分割出多个元素，则进行二次检测
            elif len(Borderlist)>1:
                predict_second = []
                for l in Borderlist: # 按前后顺序排列
                    inputs = torch.unsqueeze(l,dim=0)
                    if use_cuda:
                        inputs = inputs.cuda()
                    outputs = model(inputs)
                    outputs_p = F.softmax(outputs,dim=1)
                    _,predicted = torch.max(outputs.data,dim=1)
                    # 如果二次检测概率小于一次检测概率，则舍去该部分
                    if torch.max(outputs_p).item()<=probability[m]:
                        continue
                    # 大于一次检测概率，则使用二次检测结果替换一次检测结果
                    else:
                        predict_second.append(predicted.item())
                # 使用二次检测结果替换一次检测
                predict_list[m] = predict_second
                
        # 新的predict list
        predict_list_n = []
        for i in predict_list:
            if i==-1:
                continue
            elif isinstance(i,list):
                predict_list_n.extend(i)
            else:
                predict_list_n.append(i)
    else:
        predict_list_n = predict_list

    return predict_list_n


# 可变长度识别
def test_variable(model,test_data,test_targets,miniimg_list,use_cuda):
    targets = torch.tensor(test_targets)
    correct = 0
    total = len(test_data)
    model.eval()
    
    # analysis:还是以11个数字的测试集进行测试
    correct_list = [0 for i in range(12)]

    
    # 一次检测：
    for n in tqdm(range(len(test_data))):
        totall = len(test_data[n])
        correctt = 0
        predict_list = []
        output_pro = []
        for m in range(len(test_data[n])):
            # 增加一维：
            inputs = torch.unsqueeze(test_data[n][m],dim=0)
            if use_cuda:
                inputs = inputs.cuda()
            outputs = model(inputs)

            # softmax：将输出转为概率
            outputs_p = F.softmax(outputs,dim=1)
            # 一张图片所有数字的概率：
            output_pro.append(torch.max(outputs_p).item())

            # 预测结果
            _,predicted = torch.max(outputs.data,dim=1)
            predict_list.append(predicted.item())
        
        # 二次检测
        predict_list_n = second_test(model,output_pro,predict_list,miniimg_list[n],use_cuda)
        
        # 计算准确率:
        # 从左往右计算，和target一一对应
        for m in range(min(len(targets[n]),len(predict_list_n))):
            if targets[n][m]==predict_list_n[m]:
                correctt+=1
        # 只有当target个数和predict_list个数相同且全部正确，才能算一个图像的可变数字识别成功
        if len(targets[n])==len(predict_list_n) and correctt == totall:
            correct +=1
        
        # 统计
        correct_list[correctt]+=1
       
        if (n+1)%300==0:
            print(correct,n+1)
            acc = correct/(n+1)*100
            print("Acc in test dataset: {}%".format(acc))
        
    acc = correct/total*100
    print("Acc in test dataset: {}%".format(acc))

    single_count = 0
    for i in range(len(correct_list)):
        single_count+=i*correct_list[i]
    single_acc  = single_count/(len(test_data[0])*len(test_targets))
    print("Single Digit Acc:{}%".format(single_acc*100))

    # save picture:
    fig = plt.figure()
    plt.bar(np.arange(0,len(correct_list),1),correct_list)
    plt.title("Correct Distribution")
    plt.ylabel("Count")
    plt.xlabel("Digits")
    plt.xticks([i for i in range(0,len(correct_list),2)])
    for i,j in zip(np.arange(0,len(correct_list),1),correct_list):
        if correct_list[i] !=0:
            plt.text(i,j+1,j,ha='center')
    plt.savefig('Correct_Distribution.png')
    plt.close(fig)


    return acc,correct_list


# use handwrite img to test:
def testv2(args):
    # read:
    handwrite_data = cv2.imread(args.data_path)
    handwrite_target = os.path.basename(args.data_path)[:-4]
   
    test_data,test_target,miniimg_list = data_process(handwrite_data,handwrite_target,handwrite=True)

    if len(test_data)==0:
        return
    model = torch.load(args.weight_path,map_location='cpu') 
    
    model.eval()
   
    predict_list = []
    probability = []
    for m in range(len(test_data)):
        inputs = torch.unsqueeze(test_data[m],dim=0)
        outputs = model(inputs)
        # softmax:
        outputs_p = F.softmax(outputs,dim=1)
        probability.append(torch.max(outputs_p).item())
        _,predicted = torch.max(outputs.data,dim=1)
        predict_list.append(predicted.item())
    
    # 二次检测：
    predict_list = second_test(model,probability,predict_list,miniimg_list,use_cuda=False)

    print("predict:",end=' ')
    print(predict_list)
    print("target:",end=' ')
    test_target = [i.item() for i in test_target]
    print(test_target)



def run_experiment(args):
    # load data
    # use handwrite img to test:
    if args.handwrite:
        # check path
        if not os.path.exists(args.data_path):
            print("{} doesn't exist. Please Check!".format(args.data_path))
            return
        else:
           testv2(args)
    
    else:
        # use MNIST to generate
        if args.generate:
            generate(args)
        if not os.path.exists(args.data_path):
            print("{} doesn't exist. Please Check!".format(args.data_path))
            return
        # Read Digit Image and target:
        test_data,test_targets = Read_MultipleDigit(args.data_path)
        # segment and preprocess:
        test_data,test_targets,miniimg_list = data_process(test_data,test_targets,handwrite=False)
        if len(test_data)==0:
            return
        
        # set GPU:
        use_cuda = torch.cuda.is_available() and args.gpu>=0
        if use_cuda:
            torch.cuda.set_device(args.gpu)
            print("Running on the GPU")
        # load model:
        if use_cuda:
            model = torch.load(args.weight_path) 
            model.cuda()
        else:
            model = torch.load(args.weight_path,map_location='cpu') 
            
        test_variable(model,test_data,test_targets,miniimg_list,use_cuda)
    

if __name__ == '__main__':
    # 定义一系列参数
    parser = argparse.ArgumentParser(description='Multiple Digits Classification')
    parser.add_argument('--gpu',type=int,default=0,help='gpu')
    parser.add_argument('--handwrite',action='store_true',help='use human handwriting as test dataset')
    parser.add_argument('--generate',action='store_true',help='use MNIST dataset to generate test dataset')
    parser.add_argument('--data_path',type=str,default='./data/test',help='data path')
    parser.add_argument('--weight_path',type=str,default='./checkpoints/Resnet18_ft.pth', help='pretrained weights')
    
    args = parser.parse_args()
    # 运行实验
    run_experiment(args)




    
    




