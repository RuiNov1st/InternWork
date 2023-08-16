import torch
import argparse
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from data.data import data_process,get_transform,output_labels
from model import ResNet18

def predict(model,dataloader,use_cuda):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for _,data in tqdm(enumerate(dataloader)):
            inputs,targets = data
            if use_cuda:
                inputs,targets = inputs.cuda(),targets.cuda()

            outputs = model(inputs)
            
            _,predicted = torch.max(outputs.data,dim=1)
            total += inputs.shape[0]
            correct+=(targets==predicted).sum().item()

    acc = correct/total*100
    print("Acc in test dataset: {}%".format(acc))
    return acc


def train(model,train_dataloader,val_dataloader,args,use_cuda):
    # train ResNet18
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate)

    loss_lists = []
    acc_temp = 0
    for e in tqdm(range(args.epoch)):
        model.train()
        train_loss = 0
        total = 0
        correct = 0
        for _ ,data in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs,targets = data

            # to GPU
            if use_cuda:
                inputs,targets = inputs.cuda(),targets.cuda()
            # train
            outputs = model(inputs)
            
            loss = criterion(outputs,targets.long())
            # loss
            loss.backward()
            optimizer.step()
            
            # compute:
            train_loss += loss.item()
            _,predicted = torch.max(outputs.data,dim=1)
            total+=inputs.shape[0]
            correct+=(targets==predicted).sum().item()

        # validate
        if (e+1)%5==0:
            val_acc = predict(model,val_dataloader,use_cuda)
            # save model:
            if val_acc>acc_temp:
                acc_temp = val_acc
                # save model
                torch.save(model,args.weight_path)
            
        # save loss:
        if use_cuda:
            loss_lists.append(loss.cpu().detach().numpy())
        else:
            loss_lists.append(loss.detach().numpy())
        acc = correct/total*100
        print("epoch {}: loss {} acc {}".format(e,loss,acc))
    

    # show loss curve:
    fig = plt.figure()
    plt.plot(np.arange(0,len(loss_lists),1),loss_lists)
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig('Loss Curve.png')
    plt.close(fig)


def predict_v2(model,img):
    model.eval()
   
    inputs = torch.unsqueeze(img,dim=0)
    outputs = model(inputs)
    _,predicted = torch.max(outputs.data,dim=1)
    
    print("predict:",end=' ')
    print(output_labels[predicted])




# 主函数
def run_experiment(args):
    # 数据路径检查
    if not os.path.exists(args.data_path):
        print("Data Path not exist, please check.")
        return 
    # 手写图片测试
    if args.handdraw:
        # 读取数据
        img = cv2.imread(args.data_path)
        # process
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
        transform = get_transform()
        img = transform(img)
        # predict:
        model_ft = torch.load(args.weight_path,map_location='cpu')
        predict_v2(model_ft,img)
    else:
        # get data:
        train_dataloader,val_dataloader,test_dataloader = data_process(args)
        # set GPU:
        use_cuda = torch.cuda.is_available() and args.gpu>=0
        if use_cuda:
            torch.cuda.set_device(args.gpu)
            print("Running on the GPU")
        
        # only test:
        if args.test:
            if not os.path.exists(args.weight_path):
                print("Path not exist, please check.")
            else:
                if use_cuda:
                    model_ft = torch.load(args.weight_path)
                else:
                    model_ft = torch.load(args.weight_path,map_location='cpu')
                predict(model_ft,test_dataloader,use_cuda)
        else:
            # 全参训练
            model_ft = ResNet18().resnet18
            if use_cuda:
                model_ft.cuda()
            # 训练
            train(model_ft,train_dataloader,val_dataloader,args,use_cuda)
            # 测试
            predict(model_ft,test_dataloader,use_cuda)


if __name__ == '__main__':
    # 定义一系列参数
    parser = argparse.ArgumentParser(description='Shape Classification')
    parser.add_argument('--gpu',type=int,default=0,help='gpu')
    parser.add_argument('--batch_size',type=int,default=64,help='batch-szie')
    parser.add_argument('--epoch',type=int,default=20)
    parser.add_argument('--learning_rate',type=float,default=0.01)
    parser.add_argument('--test',action='store_true',help='directly test')
    parser.add_argument('--handdraw',action='store_true',help='use human handwriting as test dataset')
    parser.add_argument('--data_path',type=str,default='./data/dataset/labdata',help='data path')
    parser.add_argument('--weight_path',type=str,default='./checkpoints/ResNet_shape_transform.pth', help='pretrained weights')
    
    args = parser.parse_args()
    # 运行实验
    run_experiment(args)