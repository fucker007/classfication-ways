# -*- coding: utf-8 -*-
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


def VGG_16(spatial_size, classes, channels, channel_first=True, weights_path=None):
	model = Sequential()

	#224X224-->225X225
	if channel_first:
		model.add(ZeroPadding2D((1,1),input_shape=(channels, spatial_size, spatial_size)))
	else:
		model.add(ZeroPadding2D((1,1),input_shape=(spatial_size, spatial_size, channels)))

	#the input image dimension 224X224
	model.add(Conv2D(64, (3, 3), activation='relu'))#卷积后 64(feature)X 224X224(width,height for every feature).
	model.add(ZeroPadding2D((1,1)))#64X224X224-->64(feature)X225X225
	model.add(Conv2D(64, (3, 3), activation='relu')) #  64(feature)X224X224
	model.add(MaxPooling2D((2,2), strides=(2,2))) #池化后64X(feature)X224X224-->64(feature)X112X112


	model.add(ZeroPadding2D((1,1))) #112X112-->113X113
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2))) # 33

	model.add(Flatten())
	model.add(Dense(4096, activation='relu')) # 34
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu')) # 35
	model.add(Dropout(0.5))
	model.add(Dense(2622, activation='softmax')) # Dropped

	if os.path.exists("./data/Weights/0/vgg_spatial_0_CASME2_Optical_0.h5"):
		model.pop()
		model.add(Dense(classes, activation='softmax'))  # 36
		model.load_weights("./data/Weights/0/vgg_spatial_0_CASME2_Optical_0.h5")
	else:
		if weights_path:
			model.load_weights(weights_path)
		model.pop()
		model.add(Dense(classes, activation='softmax')) # 36

	return model
def myResnet50(spatial_size, classes, channels, channel_first=True, weights_path=None):

	input_tensor = Input(shape=(3,224, 224))
	base_model = resnet50.ResNet50(include_top=False,input_tensor=input_tensor, input_shape = (3,224, 224),
								   weights='imagenet',classes = 2,pooling='avg')
	for layer in base_model.layers:
		layer.trainable = False
	X = base_model.output
	predictions = Dense(2, activation='softmax')(X)
	model = Model(inputs=base_model.input, outputs=predictions)
	return model

#https://blog.csdn.net/sinat_26917383/article/details/72861152
# good paper
# resnet50 = ResNet50()
	# model = DenseNet121()
	# model = DenseNet169()
	# model = DenseNet201()
	# model = InceptionResNetV2()
	# model = InceptionV3()
	# model = MobileNet()
	# model = NASNetLarge()
	# model = VGG16()
	# model = VGG19()
	#http://www.cnblogs.com/hutao722/p/10008581.html
def mySpatialModel(model_name,spatial_size, nb_classes, channels, channel_first=True, weights_path=None,
				   lr=0.005, decay=1e-6, momentum=0.9,plot_model=True):

	input_tensor = Input(shape=(channels, spatial_size, spatial_size))
	input_shape = (channels, spatial_size, spatial_size)
	base_model=None
	predictions=None
	data_dim=1024
	if model_name=='ResNet50':

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

		base_model = kerasApp.VGG16(include_top=False, input_tensor=input_tensor, input_shape=input_shape,
									   weights=weights_path, classes=nb_classes, pooling=None)
		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)  # add a global spatial average pooling layer
		x = Dense(1024, activation='relu')(x) # let's add a fully-connected layer

		predictions = Dense(nb_classes, activation='softmax')(x) # and a logistic layer
		model = Model(inputs=base_model.input, outputs=predictions)
	elif model_name == 'VGG19':
		base_model = kerasApp.VGG19(include_top=False, input_tensor=input_tensor, input_shape=input_shape,
									weights=weights_path ,classes=2, pooling=None)

		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)

	elif model_name=='InceptionV3':
		base_model = kerasApp.InceptionV3(weights=weights_path, include_top=False, pooling=None,
								 input_shape=input_shape, classes=nb_classes)

		x = base_model.output
		# 添加自己的全链接分类层
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nb_classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
	elif model_name=='InceptionResNetV2':
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
		input_tensor = Input(shape=( spatial_size, spatial_size,channels))
		input_shape = (spatial_size, spatial_size,channels)
		base_model = kerasApp.DenseNet201(weights=weights_path, include_top=False, pooling=None,
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


	sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


	# 绘制模型
	#if plot_model:
	#	plot_model(model, to_file=model_name+'.png', show_shapes=True)
	return model


def mySpatialModelChannelTest(model_name,spatial_size, nb_classes, channels, channel_first=True, weights_path=None,
				   lr=0.005, decay=1e-6, momentum=0.9,plot_model=True):

	input_tensor = Input(shape=(channels, spatial_size, spatial_size))
	input_shape = (channels, spatial_size, spatial_size)
	base_model=None
	predictions=None
	data_dim=1024

	base_model = kerasApp.ResNet50(include_top=False, input_tensor=input_tensor, input_shape=input_shape,
									   weights=None, classes=nb_classes, pooling=None)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(nb_classes, activation='softmax')(x)
	# 训练模型
	model = Model(inputs=base_model.input, outputs=predictions)
	print_shape(model,model_name)


	base_model = kerasApp.VGG16(include_top=False, input_tensor=input_tensor, input_shape=input_shape,
								   weights=None, classes=nb_classes, pooling=None)
	x = base_model.output
	# 添加自己的全链接分类层
	x = GlobalAveragePooling2D()(x)  # add a global spatial average pooling layer
	x = Dense(1024, activation='relu')(x)  # let's add a fully-connected layer
	predictions = Dense(nb_classes, activation='softmax')(x)
	# 训练模型
	model = Model(inputs=base_model.input, outputs=predictions)
	print_shape(model, model_name)

	base_model = kerasApp.VGG19(include_top=False, input_tensor=input_tensor, input_shape=input_shape,
								weights=None, classes=2, pooling='avg')
	print_shape(base_model, model_name)
	base_model = kerasApp.InceptionV3(weights=None, include_top=False, pooling=None,
							 input_shape=input_shape, classes=nb_classes)
	print_shape(base_model, model_name)
	base_model = kerasApp.InceptionResNetV2(weights=None, include_top=False, pooling=None,
							 input_shape=input_shape, classes=nb_classes)
	x = base_model.output
	# 添加自己的全链接分类层
	x = GlobalAveragePooling2D()(x)
	predictions = Dense(nb_classes, activation='softmax')(x)
	# 训练模型
	model = Model(inputs=base_model.input, outputs=predictions)
	print_shape(model, model_name)
	#channel last
	input_tensor_Xception = Input(shape=( spatial_size, spatial_size,channels))
	input_shape__Xception = (spatial_size, spatial_size,channels)
	base_model = kerasApp.Xception(weights=None, include_top=False, pooling=None,
											input_shape=input_shape__Xception, classes=nb_classes)
	print_shape(base_model, model_name)

	base_model = kerasApp.DenseNet121(weights=None, include_top=False, pooling=None,
											input_shape=input_shape, classes=nb_classes)
	print_shape(base_model, model_name)

	base_model = kerasApp.DenseNet169(weights=None, include_top=False, pooling=None,
											input_shape=input_shape, classes=nb_classes)

	print_shape(base_model, model_name)

	base_model = kerasApp.DenseNet201(weights=None, include_top=False, pooling=None,
											input_shape=input_shape, classes=nb_classes)

	print_shape(base_model, model_name)
	input_shape = (channels, spatial_size, spatial_size)

	base_model = kerasApp.MobileNet(weights=None, include_top=False, pooling=None,
												  input_shape=input_shape, classes=nb_classes)

def print_shape(model,model_name):

	featureLayer = model.layers[len(model.layers) - 2]
	print(model_name+str(featureLayer.output_shape))
	print(featureLayer.output_shape[1])

def plot_training(self, history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(acc))
	plt.plot(epochs, acc, 'b-')
	plt.plot(epochs, val_acc, 'r')
	plt.title('Training and validation accuracy')
	plt.figure()
	plt.plot(epochs, loss, 'b-')
	plt.plot(epochs, val_loss, 'r-')
	plt.title('Training and validation loss')
	plt.show()

def temporal_module(data_dim, timesteps_TIM, classes, weights_path=None):
	model = Sequential()
	model.add(LSTM(3000, return_sequences=False, input_shape=(timesteps_TIM, data_dim)))
	#model.add(LSTM(3000, return_sequences=False))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(classes, activation='sigmoid'))

	if weights_path:
		model.load_weights(weights_path)

	return model	


def convolutional_autoencoder(classes, spatial_size, channel_first=True, weights_path=None):
	model = Sequential()

	# encoder
	if channel_first:
		model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(3, spatial_size, spatial_size), padding='same'))
	else:
		model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(spatial_size, spatial_size, 3), padding='same'))

	model.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='same'))

	# decoder
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2))	
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2))
	model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))


	return model


def VGG_16_tim(spatial_size, classes, channels, channel_first=True, weights_path=None):
	model = Sequential()
	if channel_first:
		model.add(ZeroPadding2D((1,1),input_shape=(channels, spatial_size, spatial_size)))
	else:
		model.add(ZeroPadding2D((1,1),input_shape=(spatial_size, spatial_size, channels)))


	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2))) # 33

	model.add(Flatten())
	model.add(Dense(4096, activation='relu')) # 34
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu')) # 35
	model.add(Dropout(0.5))
	model.add(Dense(2622, activation='softmax')) # Dropped


	if weights_path:
		model.load_weights(weights_path)
	model.pop()
	model.add(Dense(classes, activation='softmax')) # 36
	
	return model
