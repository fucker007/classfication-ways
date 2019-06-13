#coding:utf-8
from PIL import Image
from keras.utils import np_utils,plot_model
from keras import optimizers
import numpy as np
from keras.preprocessing import image
from keras import callbacks
import os.path
import sys
import keras
from keras import losses
import matplotlib.pyplot as plt
def get_train_all(train_path,image_size,channel):
    '''
    this fuction is for get train data and labels ,what we want to do is that we can get a matrixc of it
    if you have other train data path ,just change the follow argument train_path
    :param train_path:
    :param channel: default = 3, yes you can change it ,whatever if you want.
    :return: return a train data and labels
    '''
    train_datas = []
    train_lables = []
    print("the data of train path is : " + train_path)
    train_datas,train_lables = read_data_from_path(train_path,image_size,channel)
    return train_datas[0:100,:,:], train_lables[0:100,:]

def get_val_all(val_path,image_size,channel):
    val_datas = []
    val_lables = []
    print("the data of validation path is : " + val_path)
    val_datas,val_lables = read_data_from_path(val_path,image_size,channel)
    return val_datas, val_lables

def get_test_all(test_path,image_size,channel):
    test_datas =[]
    test_lables = []
    print ("the data of test path is : " +test_path)
    test_datas = read_data_from_path(test_path,image_size,channel)
    return test_datas, test_lables

def read_data_from_path(something_path,image_size,channel):
    '''
    this is a basic function for read image from file and translate it into arrays,
    three function will use this function ,beacuse of they have some prosses.
    :param something_path: this param may is train_path,val_path or test_path
    :param image_size: if you want change image size ,you can code on the front of this function
    :param channel: default = 3, yes you can change it ,whatever if you want.
    :return: a list shape :(*,image_size,image_size,channel)
    '''
    #print("if you want resize image size ,please input it on there ")
    #print("default = 224 :" ,end=" ")
    print("*****reading from '"+something_path+"' data **** ")
    print("      image size is %d: "%image_size)
    datas = []
    lables = []
    if os.path.exists(something_path):
        print("      file is exist ,do worry, we can start this project ...")
        #I want to code main step on there, however I got a question is that
        #how can I deal with a problem:there have more dataLabel .
        dirs = os.listdir(something_path)
        file1_path = something_path + dirs[0] + '/'
        file2_path = something_path + dirs[1] + '/'
        file1 = os.listdir(file1_path)
        file2 = os.listdir(file2_path)
        file_all = file1+file2
        file1_length = file1.__len__()
        file2_length = file2.__len__()
        #这里将所有的目录配置好，
        # # read dirs[0] get data_right,and lable_right
        print("      %s下： %d" % (str(dirs[0]),file1_length))
        data_right = []
        i = 0
        for file in file1:
            data_right.append(file1_path+""+file)
            i = i + 1
            if i >= 100:
                break
        print("         data_right[0]: %s" % data_right)
        i = 0
        data_wrong = []
        for f in file2:
            data_wrong.append(file2_path+""+f)
            i = i + 1
            if i >= 100:
                break
        print("         data_wrong[0]: %s" % data_wrong)
        #lable_right = np.ones((file1_length, 1),dtype=int)  # [0,1]
        #lable_wrong = np.zeros((file2_length, 1),dtype=int)  # [1,0]
        lable_right = np.ones((100, 1), dtype=int)  # [0,1]
        lable_wrong = np.zeros((100, 1), dtype=int)  # [1,0]
        #combin data_right and data_wrong
        data_right = np.array(data_right)
        data_wrong = np.array(data_wrong)
        datas = np.hstack((data_right,data_wrong))
        datas = np.array([datas])
        datas = datas.transpose()
        #combin lable_right and lable_wrong
        lables = np.vstack((lable_right,lable_wrong))
        print("         datas size :%d" % datas.size)
        print("         lables size :%d" % lables.size)
        # 这里的数组出来的是2行12500列，第一行是image_list的数据，第二行是label_list的数据
        temp = np.hstack((datas,lables))
        # 将其转换为2行*列，第一行是datas数据，第二行是lables数据
        #temp = temp.transpose()
        #对应的打乱顺序
        np.random.shuffle(temp)
        np.random.shuffle(temp)
        print("         train data and lable data shape:",temp.shape)
        file_path = list(temp[:,0])
        lables = list(temp[:,1])
        for i in range(lables.__len__()):
            lables[i] = int(lables[i])
        # 获取最终datas 和lables 热点化处理,
        datas = image_array_list(file_path, file_all, image_size)
        #将lable数据全部转换为int类型
        lables = [int(i) for i in lables]
        lables = np_utils.to_categorical(lables,2)
        print("         data_shape:", datas.shape)
        print("         lable_shape:", lables.shape)
    else:
        print ("      file or dir is not exists!!")
        #异常退出，
        sys.exit(0)
        np.vstack()
    print("*****read data is over ******")
    return datas,lables
def image_array_list(file_path,files,image_size):
    '''
    translate lack-data into mtric,and lables's size = file size
    :param something_path:
    :param files:
    :param image_size:
    :return: (*,image_size,image_size,3)
    '''
    image_data = []
    #for i in range(files.__len__()):
    for i in range(100):
        img = Image.open(file_path[i])
        #is rgb image？
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img.resize((image_size,image_size),Image.BILINEAR)
        img_element = image.img_to_array(img)
        image_data.append(img_element)
    data = np.array(image_data)

    return data / 255
#记录损失值和相关参数
class LossHistory(keras.callbacks.Callback):
    def __init__(self, train_path,model_name,train_id):
        self.train_path = train_path
        self.model_name = model_name
        self.train_id = train_id

    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        dirname = self.train_path+'train_procese/'+self.train_id+'/'
        filename=dirname +self.model_name+'_'+"_loss_batch.txt"
        oldFile=False
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if os.path.exists(filename):
            oldFile=True
        with open(filename,'a', encoding='utf-8') as f:
            if oldFile==False:
                f.writelines("loss,acc"+'\n')
            f.writelines(str(logs.get('loss'))+","+ str(logs.get('acc'))+'\n')

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        dirname = self.train_path + 'train_procese/' + self.train_id + '/'
        filename = dirname + self.model_name + '_' + "_loss_epoch.txt"
        oldFile = False
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if os.path.exists(filename):
            oldFile = True
        with open(filename, 'a', encoding='utf-8') as f:
            if oldFile == False:
                f.writelines("loss,acc,val_loss,val_acc"+'\n');
            f.writelines(str(logs.get('loss'))+","+ str(logs.get('acc'))+','+str(logs.get('val_loss'))+","+str(logs.get('val_acc'))+'\n')
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')

        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(self.train_path+'/train_procese/'+self.train_id+'/'+self.model_name+'_'+"loss_batch.png")
        plt.show()




#创建一个optimizer用于后面的训练中。
def getOptimizer():
    #最好优化器是4
    which_optimizer = input("please choose a optimizer,1.sgd,2.adam,3 Adagrad，4 Adadelta,5 Adamax,6 Nadam")
    if which_optimizer == '1':
        lr = 0.0001  # input("please input learning rate,I suggestion you can write in somewhere (learing rate):")
        lr = float(lr)
        decay = 1e-6 # input("please input decay,I suggestion you can write in somewhere (learing rate):")
        decay = float(decay)
        momentum = 0.9  # input("please input momentun,I suggestion you can write in somewhere (learing rate):")
        momentum = float(momentum)
        sgd = optimizers.SGD(lr, decay, momentum, nesterov=True)
        return sgd
    elif which_optimizer == '2':
        lr = 0.0001 # input("please input learning rate,I suggestion you can write in somewhere (learing rate):")
        lr = float(lr)
        decay = 1e-9  # input("please input decay,I suggestion you can write in somewhere (learing rate):")
        decay = float(decay)
        adam = optimizers.Adam(lr,decay)
        return adam
    elif which_optimizer == '3':
        Adagrad = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
        return Adagrad
    elif which_optimizer == '4':
        Adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        return  Adadelta
    elif which_optimizer == '5':
        Adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        return Adamax
    elif which_optimizer == '6':
        Nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        return Nadam
    elif which_optimizer == '7':
       RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
       return RMSprop
    else:
        print("do you really understand what my mean!!")

        sys.exit(0)
#获取loss fuction拥有训练
def getLoss():
    print("pelease input which loss function do you want to choose:")
    which_loss = input("1 mean_squared_error, 2 mean_absolute_error, 3 mean_absolute_percentage_error,"
                       " 4 mean_squared_logarithmic_error, 5 squared_hinge"
                       "6 hinge, 7 categorical_hinge,8 logcosh,9 categorical_crossentropy,10 sparse_categorical_crossentropy"
                       "11 binary_crossentropy, 12 kullback_leibler_divergence"
                       "13 poisson, 14 cosine_proximity")
    if which_loss == '1':
        return 'mean_squared_error'
    elif which_loss == '2':
        return 'mean_absolute_error'
    elif which_loss == '3':
        return 'mean_absolute_percentage_error'
    elif which_loss == '4':
        return 'mean_squared_logarithmic_error'
    elif which_loss == '5':
        return 'squared_hinge'
    elif which_loss == '6':
        return 'hinge'
    elif which_loss == '7':
        return 'categorical_hinge'
    elif which_loss == '8':
        return 'logcosh'
    elif which_loss == '9':
        return 'categorical_crossentropy' #用于多分类
    elif which_loss == '10':
        return 'sparse_categorical_crossentropy'
    elif which_loss == '11':
        return 'binary_crossentropy'  #用于二分类
    elif which_loss == '12':
        return 'kullback_leibler_divergence'
    elif which_loss == '13':
        return 'poisson'
    elif which_loss == '14':
        return 'cosine_proximity'
    else:
        print("please try it again and have a idea about loss fuction!")
        sys.exit(0)




