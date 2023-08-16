import json
import pylab as pl
import argparse
import os

parser = argparse.ArgumentParser(description='json to image')
parser.add_argument('--label',type=str,required=True,help='label')
args = parser.parse_args()

# json文件所在绝对路径
json_path = './dataset/quickdraw/{}/{}.json'.format(args.label,args.label)
# 保存的图片的路径：
image_path = './dataset/quickdraw/{}/image/'.format(args.label)
if not os.path.exists(image_path): 
      os.mkdir(image_path)

f = open(json_path)
setting = json.load(f)
for j in range(0,1000):  #转化保存1000个图
    for i in range(0,len(setting[j]['drawing'])):
            x = setting[j]['drawing'][i][0]
            y = setting[j]['drawing'][i][1]
            # f=interpolate.interp1d(x,y,kind="slinear") # 线性插值
            pl.plot(x,y,'k')
    ax = pl.gca() 
    ax.xaxis.set_ticks_position('top') 
    ax.invert_yaxis()
    pl.axis('off')
    pl.savefig(os.path.join(image_path,'{}.png'.format(j)))  # 保存位置
    pl.close()
