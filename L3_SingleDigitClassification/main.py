import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import ResNet18
import argparse
from data.data import get_MNIST,get_dataloader
import os

# Finetune ResNet18
def finetune(model,train_dataloader,args):
    model.train() # 模型设置为训练状态
    criterion = torch.nn.CrossEntropyLoss() # loss函数：交叉熵损失，用于分类任务
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate) # 优化器：默认Adam

    loss_lists = []
    for e in tqdm(range(args.epoch)):
        train_loss = 0
        total = 0
        correct = 0
        for _ ,data in enumerate(train_dataloader):
            optimizer.zero_grad() # 梯度归零
            inputs,targets = data
            # to GPU
            inputs,targets = inputs.cuda(),targets.cuda()
            # train
            outputs = model(inputs)
            loss = criterion(outputs,targets) # 计算loss
            # loss
            loss.backward() # loss反向传播
            optimizer.step() # 参数更新
            
            # compute:
            train_loss += loss.item()
            _,predicted = torch.max(outputs.data,dim=1)
            total+=inputs.shape[0]
            correct+=(targets==predicted).sum().item()
            
            
        # save loss:
        loss_lists.append(loss.cpu().detach().numpy())
        acc = correct/total*100
        print("epoch {}: loss {} acc {}".format(e,loss,acc))
    
    # save model
    torch.save(model,args.save_model_path) # 模型保存
    print("Save Model in {} Successfully!".format(args.save_model_path))

     # show loss curve:
    if args.loss_img:
        fig = plt.figure()
        plt.plot(np.arange(0,len(loss_lists),1),loss_lists)
        plt.title("Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig("finetune_loss.png")
        plt.close(fig)

# test:
def predict(model,test_dataloader):
    # predict:
    correct = 0
    total = 0
    model.eval() # 模型设置为推理状态
    with torch.no_grad(): # 测试时无需梯度计算
        for data in tqdm(test_dataloader):
            inputs,targets = data
            
            inputs,targets = inputs.cuda(),targets.cuda()
            outputs = model(inputs)
            
            _,predicted = torch.max(outputs.data,dim=1)
            total += inputs.shape[0]
            correct+=(targets==predicted).sum().item()
            
    acc = correct/total*100
    print("Acc in test dataset: {}%".format(acc))



def run_experiment(args):
    # get model
    model = ResNet18(args.weight_path)

    # get data
    if args.dataset=='MNIST':
        train_dataset,test_dataset = get_MNIST(args,model.transform)
    train_dataloader,test_dataloader = get_dataloader(train_dataset,test_dataset,args)

    # set GPU
    use_cuda = torch.cuda.is_available() and args.gpu>=0
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        print("Running on the GPU")
    
    # only test
    if args.test and os.path.exists(args.save_model_path):
        resnet18_ft = torch.load(args.save_model_path) 
        resnet18_ft.cuda()
        predict(resnet18_ft,test_dataloader)
    elif args.test and not os.path.exists(args.save_model_path):
        print("{} not exist, please check.".format(args.save_model_path))
    # finetune
    else:
        model.finetune_modify()
        resnet18 = model.resnet18
        resnet18.cuda()
        finetune(resnet18,train_dataloader,args)
        

# 训练命令实例：
# python main.py --gpu 0 --batch_size 64 --dataset MNIST --data_path ./data/ --weight_path ./checkpoints/resnet18-f37072fd.pth --save_model_path ./checkpoints/resnet18_ft.pth --epoch 10 --learning_rate 0.001 
if __name__ == '__main__':
    # 定义一系列参数
    parser = argparse.ArgumentParser(description='ResNet18')
    parser.add_argument('--gpu',type=int,default=0,help='gpu')
    parser.add_argument('--batch_size',type=int,default=64,help='batch-szie')
    parser.add_argument('--dataset',type=str,default='MNIST',help='dataset')
    parser.add_argument('--data_path',type=str,default='./data/',help='data path')
    parser.add_argument('--weight_path',type=str,default='./checkpoints/resnet18-f37072fd.pth', help='pretrained weights')
    parser.add_argument('--save_model_path',type=str,default='./checkpoints/resnet18_ft.pth', help='save model after finetuning')
    parser.add_argument('--epoch',type=int,default=10)
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--test',action='store_true',default=False,help='directly test without finetune')
    parser.add_argument('--loss_img',action='store_true',default=False,help='draw loss function in finetune')
    
    args = parser.parse_args()
    # 运行实验
    run_experiment(args)

    