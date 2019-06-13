#coding:utf-8
from keras.preprocessing.image import ImageDataGenerator
import os
def generate_batch_data(path):
    datagen = ImageDataGenerator(
            rotation_range=40,                                #  旋转范围
            width_shift_range=0.2,                          #  宽度调整范围
            height_shift_range=0.2,                         #  高度调整范围
            rescale=1./255,                                      #  尺度调整范围
            shear_range=0.2,                                  #  弯曲调整范围
            zoom_range=0.2,                                  #  缩放调整范围
            horizontal_flip=True,                              #  水平调整范围
            #brightness_range=0.3,                           #  亮度调整范围
            featurewise_center=True,                     #  是否特征居中
            featurewise_std_normalization=True,   #  特征是否归一化
            zca_whitening=True)                             #  是否使用 ZCA白化
    #        fill_mode='nearest')                               #  填充模式(图片大小不够时)
    return datagen.flow_from_directory(
            path,
            target_size=(224, 224),
            color_mode='rgb',
            batch_size=16)

fileNum = 0

def get_steps_per_epoch(path):
    fileNum = len([file for dir in os.listdir(path) for file in os.listdir(path+'/'+dir)])
    batch_size = 16
    steps_per_epoch=int(fileNum/batch_size)
    return steps_per_epoch