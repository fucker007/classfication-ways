# -*- coding: utf-8 -*-
#from __builtin__ import True

import numpy as np
import os

from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D
from keras.optimizers import SGD

from keras.applications import resnet50
from keras.layers import Flatten, Dense, Input
import  matplotlib as plt
import  keras.applications as kerasApp
from keras.utils import np_utils, plot_model


def mySpatialModel(model_name,spatial_size, nb_classes, channels,  weights_path=None):

	input_tensor = Input(shape=(channels, spatial_size, spatial_size))
	input_shape = (channels,spatial_size, spatial_size)
	base_model=None
	predictions=None
	data_dim=1024
	if model_name=='ResNet50':

		input_tensor = Input(shape=(spatial_size, spatial_size,channels))
		input_shape = (spatial_size, spatial_size,channels)

		base_model = kerasApp.ResNet50(include_top=False, input_tensor=input_tensor, input_shape=input_shape,
									   weights=weights_path, classes=nb_classes, pooling=None)
		x = base_model.output
		# 添加自己的全链接分类层 method 1
		#x = Flatten()(x)
		#predictions = Dense(nb_classes, activation='softmax')(x)
		#method 2
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
	elif model_name=='VGG16':
                input_tensor = Input(shape=(spatial_size,spatial_size,channels))
                input_shape = (spatial_size, spatial_size,channels)
                base_model = kerasApp.VGG16(include_top=False, input_tensor=input_tensor, input_shape=input_shape,weights=weights_path, classes=nb_classes, pooling=None)
                x = base_model.output
                x = GlobalAveragePooling2D()(x)  # add a global spatial average pooling layer
                x = Dense(1024, activation='relu')(x) # let's add a fully-connected layer
                predictions = Dense(nb_classes, activation='softmax')(x) # and a logistic layer
                model = Model(inputs=base_model.input, outputs=predictions)
	elif model_name == 'VGG19':
		input_tensor = Input(shape=(spatial_size, spatial_size,channels))
		input_shape = (spatial_size, spatial_size,channels)
		base_model = kerasApp.VGG19(include_top=False, input_tensor=input_tensor, input_shape=input_shape,
									weights=weights_path ,classes=2, pooling=None)

		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)

	elif model_name=='InceptionV3':
		input_tensor = Input(shape=( spatial_size, spatial_size,channels))
		input_shape = (spatial_size, spatial_size,channels)
		base_model = kerasApp.InceptionV3(weights=weights_path, include_top=False, pooling=None,
								 input_shape=input_shape, classes=nb_classes)

		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
	elif model_name=='InceptionResNetV2':
		input_tensor = Input(shape=( spatial_size, spatial_size,channels))
		input_shape = (spatial_size, spatial_size,channels,)
		base_model = kerasApp.InceptionResNetV2(weights=weights_path, include_top=False, pooling=None,
								 input_shape=input_shape, classes=nb_classes)

		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)
		data_dim = 1536
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
	elif model_name == 'Xception':
		input_shape_xception = (spatial_size, spatial_size,channels)

		base_model = kerasApp.Xception(weights=weights_path, include_top=False, pooling="avg",
												input_shape=input_shape_xception, classes=nb_classes)
		x = base_model.output
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)

	elif model_name == 'DenseNet121':
		input_tensor = Input(shape=(spatial_size, spatial_size, channels))
		input_shape = (spatial_size, spatial_size, channels)
		base_model = kerasApp.DenseNet121(weights=weights_path, include_top=False, pooling=None,
												input_shape=input_shape, classes=nb_classes)

		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)

		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
	elif model_name == 'DenseNet169':
		base_model = kerasApp.DenseNet169(weights=weights_path, include_top=False, pooling=None,
												input_shape=input_shape, classes=nb_classes)

		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)

		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
	elif model_name == 'DenseNet201':
		input_tensor = Input(shape=(spatial_size, spatial_size, channels))
		input_shape = (spatial_size, spatial_size, channels)
		base_model = kerasApp.DenseNet201(weights=weights_path, input_tensor=input_tensor,include_top=False, pooling=None,
												input_shape=input_shape, classes=nb_classes)

		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
	elif model_name == 'MobileNet':
		base_model = kerasApp.MobileNet(weights=weights_path, include_top=False, pooling=None,
										  input_shape=input_shape, classes=nb_classes)
		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		x = Dense(1024, activation='relu')(x)
		x = Dense(512, activation='relu')(x)
		data_dim=512
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
	else:
		print("this model--["+model_name+"]-- doesnt exist!")

	# 冻结base_model所有层，这样就可以正确获得bottleneck特征
	for layer in base_model.layers:
		layer.trainable = True
	# 训练模型
	model = Model(inputs=base_model.input, outputs=predictions)

	print('-------------当前base_model模型[' + model_name + "]-------------------\n")
	print('base_model层数目:' + str(len(base_model.layers)))
	print('model模型层数目:' + str(len(model.layers)))
	featureLayer=model.layers[len(model.layers)-2]
	print(featureLayer.output_shape)
	print("data_dim:" + str(featureLayer.output_shape[1]))
	print("---------------------------------------------\n")


	#sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)


	# 绘制模型
	#if plot_model:
	#	plot_model(model, to_file=model_name+'.png', show_shapes=True)
	return model
