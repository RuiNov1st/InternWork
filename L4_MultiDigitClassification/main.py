import torch
import argparse
from tqdm import tqdm
import os
import cv2
from data.data import Read_MultipleDigit,data_process
from data.generate import generate
import matplotlib.pyplot as plt
import numpy as np


def test(model,test_data,test_targets,use_cuda):
    model.eval()
    correct = 0
    total = len(test_data)

    # analysis:
    correct_list = [0 for i in range(len(test_data[0])+1)]

    for n in tqdm(range(len(test_data))):
        total_single = len(test_data[n])
        correct_single = 0
        predict_list = []
        for m in range(len(test_data[n])):
            # 增加一维：
            inputs = torch.unsqueeze(test_data[n][m],dim=0)
            target = torch.unsqueeze(test_targets[n][m],dim=0)

            if use_cuda:
                inputs,target = inputs.cuda(),target.cuda()
            outputs = model(inputs)

            _,predicted = torch.max(outputs.data,dim=1)
            predict_list.append(predicted.item())
            # 单个数字正确：
            if target==predicted:
                correct_single+=1

        # 所有数字正确
        if correct_single == total_single:
            correct +=1
    
        correct_list[correct_single]+=1

        if n%300==0:
            acc = correct/(n+1)*100
            print("Acc in test dataset: {}%".format(acc))

    # analysis:
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

# use handwrite img to test:
def testv2(args):
    # read:
    handwrite_data = cv2.imread(args.data_path)
    handwrite_target = os.path.basename(args.data_path)[:-4]
    test_data,test_target = data_process(handwrite_data,handwrite_target,True)
    if len(test_data)==0:
        return
    model = torch.load(args.weight_path,map_location='cpu') 
    
    model.eval()
    with torch.no_grad(): # 测试时无需梯度计算
        predict_list = []
        for m in range(len(test_data)):
            inputs = torch.unsqueeze(test_data[m],dim=0)
            outputs = model(inputs)
            _,predicted = torch.max(outputs.data,dim=1)
            predict_list.append(predicted.item())

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
    # use MNIST to generate
    else:
        if args.generate:
            generate(args)
        if not os.path.exists(args.data_path):
            print("{} doesn't exist. Please Check!".format(args.data_path))
            return
        # Read Digit Image and target:
        test_data,test_targets = Read_MultipleDigit(args.data_path)
        # segment and preprocess:
        test_data,test_targets = data_process(test_data,test_targets)
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

        # test
        test(model,test_data,test_targets,use_cuda)

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




    
    




