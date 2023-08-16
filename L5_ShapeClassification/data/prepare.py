import imageio
import numpy as np
import os
from pathlib import Path
from six.moves import cPickle as pickle
import ntpath
from sklearn.model_selection import train_test_split
import cv2


"""
准备数据流程：
从源网站下载到本地后，原始数据经过的一系列处理流程，最后得到train/val/test三个npz文件用于模型训练和测试。
1. 数据集来源：
HDS（https://github.com/frobertpixto/hand-drawn-shapes-dataset/tree/main）
quickdraw（https://quickdraw.withgoogle.com/data）
2. 数据集处理流程：
黑白反向
尺寸放缩为224*224
处理为circle/rectangle/triangle/star4类
train/valid/test比例分配
"""

# path configuration:
HDS_BASEDIR = 'dataset/hand-drawn-shapes-dataset'
HDS_DATA_DIR = os.path.join(HDS_BASEDIR,'data')
HDS_PICKLE_DIR = os.path.join(HDS_BASEDIR,'pickles')
HDS_TRAIN_DATAFILE = os.path.join(HDS_PICKLE_DIR,'train.pickle')
HDS_VAL_DATAFILE = os.path.join(HDS_PICKLE_DIR,'val.pickle')
HDS_TEST_DATAFILE = os.path.join(HDS_PICKLE_DIR,'test.pickle')
HDS_STAR_DATAFILE = os.path.join(HDS_PICKLE_DIR,'star.pickle')

HDS_output_labels = [
'other',     #    0
'ellipse',   #    1
'rectangle', #    2
'triangle']  #    3


quickdraw_BASEDIR = 'dataset/quickdraw'
quickdraw_DATA_DIR= os.path.join(quickdraw_BASEDIR,'data')

quickdraw_output_labels = [
    'circle',   #    0
    'rectangle', #    1
    'triangle', # 2
    'star']  # 3

mix_BASEDIR='dataset/data'



class HDS_preprocess:
    def __init__(self):
        self.pixel_depth = 255.0
        
    def get_label_for_shape(self,shape_dir):
        shape = os.path.basename(shape_dir)
        if shape == "other":
            return 0
        elif shape == "ellipse":
            return 1
        elif shape == "rectangle":
            return 2
        elif shape == "triangle":
            return 3
        else:
            raise Exception('Unknown shape: %s' % shape)
    
    # Functions for getting array of directory paths and array of file paths
    def get_dir_paths(self,root):
        return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, n))]

    def get_file_paths(self,root):
        return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isfile(os.path.join(root, n))]

    def path_leaf(self,path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)
    
    # Normalize image by pixel depth by making it white on black instead of black on white
    def normalize_image(self,image_file):
        try:
            array = imageio.imread(image_file)
        except ValueError:
            raise

        # return 1.0 - (array.astype(float))/self.pixel_depth # (1 - x) will make it white on black
        return 255 - array
    
    def save_to_pickle(self,pickle_file, object, force=True):
        """
        Save an object to a pickle file
        """       
        if os.path.exists(pickle_file) and not force:
            print(f'{pickle_file} already present, skipping pickling')
        else:
            try:
                with open(pickle_file, 'wb') as file:
                    pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f'Unable to save object to {pickle_file}: {e}')
                raise
    
    def load_images_for_shape(self,shape_directory, pixel_depth, user_images,
                          user_images_label, label, verbose=False, min_nimages=1):
        """
        Load all images for a specific user and shape
        """      

        if verbose:
            print("directory for load_images_for_shapes: ", shape_directory)

        image_files = self.get_file_paths(shape_directory)
        image_index = 0

        for image_file in image_files:
            try:
                if self.path_leaf(image_file).startswith('.'):  # skip files like .DSStore
                    continue

                image_data_all_channels = self.normalize_image(image_file)
                image_data = image_data_all_channels[:, :, 0]

                user_images.append(image_data)
                user_images_label.append(label)
                image_index += 1
            except Exception as error:
                print(error)
                print('Skipping because of not being able to read: ', image_file)

        if image_index < min_nimages:
            raise Exception('Fewer images than expected: %d < %d' % (image_index, min_nimages))
    

    def load_images_for_user(self,user_directory, pixel_depth,
                         user_images, user_images_label,
                            verbose=False):
        """
        Load all images for a specific user
        """      
        
        images_dir = os.path.join(user_directory, "images")

        if verbose:
            print("directory for load_images_for_shapes: ", images_dir)

        shape_dirs = self.get_dir_paths(images_dir)
        for dir in shape_dirs:
            label = self.get_label_for_shape(dir)
            if label >= 0:
                self.load_images_for_shape(dir, pixel_depth, user_images, user_images_label, label)
    
    def save_dataset(self,data_paths,users,save_path):
        user_images = []
        user_images_label = []
        for user_dir in data_paths:
            user_id = user_dir[-3:] # User unique id is the last 3 letters
            if user_id in users:
                self.load_images_for_user(user_dir, self.pixel_depth, user_images, user_images_label)
        
        data = np.array(user_images)
        labels = np.array(user_images_label)
        # Save train data to single pickle file
        self.save_to_pickle(
            save_path,
            {
                'data': data,
                'labels': labels
            }
        )
        print('Pickle saved in: ', save_path)

    # 选出星星图片：
    def save_star(self,data_path,output_labels,save_path):
        # 提前检查挑选出的文件名：
        starfilelist = ['other.if2.0490',
            'other.if2.0491',
            'other.if2.0492',
            'other.if2.0493',
            'other.if2.0494',
            'other.if2.0495',
            'other.if2.0496',
            'other.if2.0497',
            'other.il1.0016',
            'other.il1.0133',
            'other.il1.1076',
            'other.il1.1077',
            'other.il1.1078']
        starpath_list = []
        for l in starfilelist:
            user_name = l.split('.')[1]
            star_path = os.path.join(data_path,'user.{}'.format(user_name),'images','other',l+'.png')
            starpath_list.append(star_path)
        
        # 保存星星图像至pickel路径
        star_array = []
        for image_file in starpath_list:
            array = cv2.imread(image_file)
            image_data_all_channels = 255 - array # 黑底白字
            image_data = cv2.cvtColor(image_data_all_channels, cv2.COLOR_BGR2GRAY) # 单通道
            star_array.append(image_data)

        star_array = np.array(star_array)
        star_label = np.array([output_labels.index('star')]*len(star_array))
        # Save train data to single pickle file
        self.save_to_pickle(
            save_path,
            {
                'data': star_array,
                'labels': star_label
            }
        )
        print('Pickle saved in: ',save_path)


def get_HDS_dataset():
    # Create Pickle directory
    Path(HDS_PICKLE_DIR).mkdir(parents=True, exist_ok=True)

    # process method:
    preprocessor = HDS_preprocess()

    # Define the Users for the validation set and the test set
    # The Rest will go in the Training set
    # The following definitions are arbitrary, but this is balanced and works well.
    validation_users = ['crt', 'il1', 'lts', 'mrt', 'nae']
    test_users =  ['u01', 'u17', 'u18', 'u19']
    data_paths = preprocessor.get_dir_paths(HDS_DATA_DIR)
    train_users = []
    for user_dir in data_paths:
        user_id = user_dir[-3:] # User unique id is the last 3 letters
        if user_id not in validation_users and user_id not in test_users:
            train_users.append(user_id)

    
    # Save training set：
    preprocessor.save_dataset(data_paths,train_users,HDS_TRAIN_DATAFILE)
    # Save validation set:
    preprocessor.save_dataset(data_paths,validation_users,HDS_VAL_DATAFILE)
    # Save testing set:
    preprocessor.save_dataset(data_paths,test_users,HDS_TEST_DATAFILE)

    # save star
    preprocessor.save_star(HDS_DATA_DIR,quickdraw_output_labels,HDS_STAR_DATAFILE)


class quickdraw_preprocess:
    def get_label_path(self,label,data_path):
        label_path = []
        path = os.path.join(data_path,label)
        for dir in  os.listdir(path):
            if os.path.isdir(os.path.join(path,dir)):
                filelist = os.listdir(os.path.join(path,dir))
                for file in filelist:
                    label_path.append(os.path.join(path,dir,file))
        return sorted(list(set(label_path)))
    

    def train_val_test_split(self,label,data_path,train_test_factor = 1/20,train_val_factor=3/19):
        # 得到该label下所有的图片文件路径
        label_path = self.get_label_path(label,data_path)
        # 按比例分配数据集：16:3:1
        x_train,x_test = train_test_split(label_path,test_size=train_test_factor,shuffle=True)
        x_train,x_val = train_test_split(x_train,test_size=train_val_factor,shuffle=True)
    
        return x_train,x_val,x_test
    
    def load_save_data(self,data,output_labels,label,save_path,name):
        # train:
        array = []
        for image_file in data:
            arr = cv2.imread(image_file)
            image_data_all_channels = 255 - arr # 黑底白字
            image_data = cv2.cvtColor(image_data_all_channels, cv2.COLOR_BGR2GRAY) # 单通道
            array.append(image_data)
        
        array = np.array(array)
        labels = np.array([output_labels.index(label)]*len(array))

        # 先存为npz文件
        np.savez(os.path.join(save_path,label,'{}_{}'.format(label,name)),x=array,y=labels)

    # 存为数据集
    def make_dataset(self,data_path,label,output_labels,save_path):
        # 先划分数据集
        x_train,x_val,x_test = self.train_val_test_split(label,data_path)
        # 保存数据
        # train:
        self.load_save_data(x_train,output_labels,label,save_path,'train')
        # valid:
        self.load_save_data(x_val,output_labels,label,save_path,'valid')
        # test:
        self.load_save_data(x_test,output_labels,label,save_path,'test')
        print("{} save successfully!".format(label))



def get_quickdraw_dataset():
    Path(quickdraw_DATA_DIR).mkdir(parents=True, exist_ok=True)
    preprocessor = quickdraw_preprocess()
    for label in quickdraw_output_labels:
        if not os.path.exists(os.path.join(quickdraw_DATA_DIR,label)):
            os.mkdir(os.path.join(quickdraw_DATA_DIR,label))
        preprocessor.make_dataset(quickdraw_BASEDIR,label,quickdraw_output_labels,quickdraw_DATA_DIR)



class mix_preprocess:
    def label_map(self,old_label):
        if old_label == 1:
            return quickdraw_output_labels.index('circle')
        elif old_label == 2:
            return quickdraw_output_labels.index('rectangle')
        elif old_label == 3:
            return quickdraw_output_labels.index('triangle')
    
    def hds_modify(self,data,label):
        new_data = []
        new_label = []
        for i in range(data.shape[0]):
            # other
            if label[i]==0:
                continue
            else:
                new_data.append(data[i])
                new_label.append(self.label_map(label[i]))
        new_data = np.array(new_data)
        new_label = np.array(new_label)
        return new_data,new_label
    
    def quickdraw_dataload(self,output_labels,data_path,name):
        array = []
        labels = []
        for label in output_labels:
            data =  np.load(os.path.join(data_path,label,'{}_{}.npz'.format(label,name)))
            data_X = data['x']
            data_y = data['y']
            array.extend(data_X)
            labels.extend(data_y)
        return array,labels
    
            
    

def mix_dataset():
    # HDS:
    # 70*70的读取，去除other，更改label
    with open(HDS_TRAIN_DATAFILE, 'rb') as file:
        HDS_train_dict = pickle.load(file)
    with open(HDS_VAL_DATAFILE, 'rb') as file:
        HDS_val_dict = pickle.load(file)
    with open(HDS_TEST_DATAFILE, 'rb') as file:
        HDS_test_dict = pickle.load(file)
    with open(HDS_STAR_DATAFILE,'rb') as file:
        HDS_star_dict = pickle.load(file)
        

    HDS_train_X = HDS_train_dict['data']
    HDS_train_y = HDS_train_dict['labels']

    HDS_val_X = HDS_val_dict['data']
    HDS_val_y = HDS_val_dict['labels']

    HDS_test_X = HDS_test_dict['data']
    HDS_test_y = HDS_test_dict['labels']

    star_X = HDS_star_dict['data']
    star_y = HDS_star_dict['labels']

    
    preprocessor = mix_preprocess()
    # train:
    HDS_train_X,HDS_train_y = preprocessor.hds_modify(HDS_train_X,HDS_train_y)
    # add star
    HDS_train_X = np.append(HDS_train_X,star_X,axis=0)
    HDS_train_y = np.append(HDS_train_y,star_y,axis=0)
    # valid and test:
    # hds modify:
    HDS_val_X,HDS_val_y = preprocessor.hds_modify(HDS_val_X,HDS_val_y)
    HDS_test_X,HDS_test_y = preprocessor.hds_modify(HDS_test_X,HDS_test_y)

    # 统一至224：
    # 直接cv2.resize统一到224：
    HDS_train_X = [cv2.resize(i,(224,244)) for i in HDS_train_X]
    HDS_val_X = [cv2.resize(i,(224,244)) for i in HDS_val_X]
    HDS_test_X = [cv2.resize(i,(224,244)) for i in HDS_test_X]

    # to array:
    HDS_train_X,HDS_val_X,HDS_test_X = np.array(HDS_train_X),np.array(HDS_val_X),np.array(HDS_test_X)

    # quickdraw:
    # 加入新的数据：
    quickdraw_train,quickdraw_val,quickdraw_test = [],[],[]
    quickdraw_train_label,quickdraw_val_label,quickdraw_test_label = [],[],[]
    # train:
    quickdraw_train,quickdraw_train_label = preprocessor.quickdraw_dataload(quickdraw_output_labels,quickdraw_DATA_DIR,'train')
    # valid:
    quickdraw_val,quickdraw_val_label = preprocessor.quickdraw_dataload(quickdraw_output_labels,quickdraw_DATA_DIR,'valid')
    # test:
    quickdraw_test,quickdraw_test_label=preprocessor.quickdraw_dataload(quickdraw_output_labels,quickdraw_DATA_DIR,'test')
    
    # cv2.resize统一到224:
    quickdraw_train = [cv2.resize(i,(224,244)) for i in quickdraw_train]
    quickdraw_val = [cv2.resize(i,(224,244)) for i in quickdraw_val]
    quickdraw_test = [cv2.resize(i,(224,244)) for i in quickdraw_test]

    quickdraw_train,quickdraw_val,quickdraw_test = np.array(quickdraw_train),np.array(quickdraw_val),np.array(quickdraw_test)
    quickdraw_train_label,quickdraw_val_label,quickdraw_test_label = np.array(quickdraw_train_label),np.array(quickdraw_val_label),np.array(quickdraw_test_label)

    
    # 合并起来：
    train_X = np.append(HDS_train_X,quickdraw_train,axis=0)
    train_y = np.append(HDS_train_y,quickdraw_train_label,axis=0)
    val_X = np.append(HDS_val_X,quickdraw_val,axis=0)
    val_y = np.append(HDS_val_y,quickdraw_val_label,axis=0)
    test_X = np.append(HDS_test_X,quickdraw_test,axis=0)
    test_y = np.append(HDS_test_y,quickdraw_test_label,axis=0)

    # 保存：
    if not os.path.exists(mix_BASEDIR):
        os.mkdir(mix_BASEDIR)
    np.savez(os.path.join(mix_BASEDIR,'train'),x=train_X,y=train_y)
    np.savez(os.path.join(mix_BASEDIR,'val'),x=val_X,y=val_y)
    np.savez(os.path.join(mix_BASEDIR,'test'),x=test_X,y=test_y)



if __name__ == '__main__':
    get_HDS_dataset()
    get_quickdraw_dataset()
    mix_dataset()



    








