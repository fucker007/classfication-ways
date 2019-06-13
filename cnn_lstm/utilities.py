# coding=utf-8
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, Model

import keras
import pydot, graphviz
from keras.utils import np_utils, plot_model

from src.cnn_lstm.reordering import readinput
from src.cnn_lstm.evaluationmatrix import fpr
import itertools
from src.cnn_lstm.pynvml import *

def Read_Input_Images(inputDir,  resizedFlag,  spatial_size, channel):
	r = w = spatial_size
	SubperdB = []
	for i in range(1,2):
		sub="jpg"+str(i)
		path = inputDir + sub + '/'  # image loading path
		imgList = readinput(path)
		numFrame = len(imgList)
		if resizedFlag == 1:
			col = w
			row = r
		else:
			img = cv2.imread(imgList[0])
			[row,col,_l] = img.shape
		for var in range(numFrame):
			img = cv2.imread(imgList[var])
			[_,_,dim] = img.shape
			if channel == 1:
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			if resizedFlag == 1:
				img = cv2.resize(img, (col,row))
			if var == 0:
				FrameperVid = img.flatten()
			else:
				FrameperVid = np.vstack((FrameperVid,img.flatten()))
		SubperdB.append(FrameperVid)
	return SubperdB

#读取批量图像
def Read_Batch_Input_Images(sub,inputDir,  resizedFlag,  spatial_size, channel):
	r = w = spatial_size
	SubperdB = []
	sub="jpg"+str(sub)
	path = inputDir + sub + '/'  # image loading path
	imgList = readinput(path)
	numFrame = len(imgList)
	if resizedFlag == 1:
		col = w
		row = r
	else:
		img = cv2.imread(imgList[0])
		[row,col,_l] = img.shape
	for var in range(numFrame):
		img = cv2.imread(imgList[var])
		[_,_,dim] = img.shape
		if channel == 1:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if resizedFlag == 1:
			img = cv2.resize(img, (col,row))
		if var == 0:
			FrameperVid = img.flatten()
		else:
			FrameperVid = np.vstack((FrameperVid,img.flatten()))
	SubperdB.append(FrameperVid)
	return SubperdB
def label_matching(labelPath,test_count):

	label_all=[]
	for i in range(1, test_count+1):
		fileName = labelPath+"/dataLabel" + str(i) + 'deep.txt'
		label_page=[]
		with open(fileName, 'r')as df:
			for line in df:
				# 如果换行符就跳过，这里用'\n'的长度来找空行
				if line.count('\n') == len(line):
					continue
				# 对每行清除前后空格（如果有的话），然后用"："分割
				for kv in [line.strip().split(' ')]:
					if(len(kv)<2):
						print("lable_match-->"+fileName+str(len(kv)) + " where has this wrong: " + line)
					else:
						label_page.append(kv[1])

		label_all.append(label_page)
	return label_all
#loading data
def data_loader_with_LOSO(subject, SubjectPerDatabase, y_labels, subjects, n_exp):

	Train_X = []
	Train_Y = []
	label_len=len(y_labels[subject])
	Test_X = np.array(SubjectPerDatabase[0])
	Test_X=Test_X[:label_len]

	Test_Y = np_utils.to_categorical(y_labels[subject], n_exp)
	Test_Y_gt = y_labels[subject]
	########### Leave-One-Subject-Out ###############

	Train_X.append(Test_X)
	Train_Y.append(y_labels[subject])
	##################################################


	############ Conversion to numpy and stacking ###############
	Train_X=np.vstack(Train_X)
	Train_Y=np.hstack(Train_Y)
	Train_Y=np_utils.to_categorical(Train_Y, n_exp)
	#############################################################

	return Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt
#这是刘洋从新写的读取代码的函数。
def data_loader_with_LOSO_modify_by_liuyang_for_train(subject, SubjectPerDatabase, y_labels, subjects, n_exp):
	Train_X = []
	Train_Y = []
	label_len = len(y_labels[subject])
	Test_X = np.array(SubjectPerDatabase[0])
	Test_X = Test_X[:label_len]
	Test_Y = np_utils.to_categorical(y_labels[subject], n_exp)
	Test_Y_gt = y_labels[subject]


def duplicate_channel(X):

	X = np.repeat(X, 3, axis=3)
	# np.set_printoptions(threshold=np.nan)
	# print(X)
	print(X.shape)

	return X

def record_scores(workplace, dB, ct, sub, order, tot_mat, n_exp, subjects):


	with open(workplace+'result'+'/sub_CT.txt','a') as csvfile:
			thewriter=csv.writer(csvfile, delimiter=' ')
			thewriter.writerow('Sub ' + str(sub+1))
			thewriter=csv.writer(csvfile,dialect=csv.excel_tab)
			for row in ct:
				thewriter.writerow(row)
			thewriter.writerow(order)
			thewriter.writerow('\n')

	if sub==subjects-1:
			# compute the accuracy, F1, P and R from the overall CT
			microAcc=np.trace(tot_mat)/np.sum(tot_mat)
			[f1,p,r]=fpr(tot_mat,n_exp)
			print(tot_mat)
			print("F1-Score: " + str(f1))
			# save into a .txt file
			with open(workplace+'result'+'/final_CT.txt','w') as csvfile:
				thewriter=csv.writer(csvfile,dialect=csv.excel_tab)
				for row in tot_mat:
					thewriter.writerow(row)

				thewriter=csv.writer(csvfile, delimiter=' ')
				thewriter.writerow('micro:' + str(microAcc))
				thewriter.writerow('F1:' + str(f1))
				thewriter.writerow('Precision:' + str(p))
				thewriter.writerow('Recall:' + str(r))



def filter_objective_samples(table): # this is to filter data with objective classes which is 1-5, omitting 6 and 7
	list_samples = []
	sub = table[0, :, 0]
	vid = table[0, :, 1]
	# print(sub)
	# print(vid)

	for count in range(len(sub)):
		pathname = 0
		if len(sub[count]) == 2:
			pathname = "sub" + sub[count] + "/" + vid[count]
		else:
			pathname = sub[count] + "/" + vid[count]
		# pathname = inputDir + pathname
		list_samples += [pathname]

	# print(list_samples)

	return list_samples



class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []  #损失率
		self.accuracy = [] #精度
		self.epochs = [] # 轮多少回
	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accuracy.append(logs.get('categorical_accuracy'))
		self.epochs.append(logs.get('epochs'))


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def record_loss_accuracy(db_home, train_id, db, history_callback):

	path=db_home + 'dataLabel/' + 'Result/' + 'loss_' + str(train_id) + '.txt'
	print(path)
	file_loss = open(path, 'a')

	file_loss.write(str(history_callback.losses) + "\n")
	file_loss.close()

	file_loss = open(db_home + 'dataLabel/' + 'Result/' + 'accuracy_' + str(train_id) + '.txt', 'a')
	file_loss.write(str(history_callback.accuracy) + "\n")
	file_loss.close()

	file_loss = open(db_home + 'dataLabel/' + 'Result/'+ 'epoch_' + str(train_id) +  '.txt', 'a')
	file_loss.write(str(history_callback.epochs) + "\n")
	file_loss.close()

def record_weights(model,flag):
	if flag == 's' or flag == 'st':
		layerOrder=len(model.layers)-2
		model = Model(inputs=model.input, outputs=model.layers[layerOrder].output)
		#plot_model(model, to_file = "spatial_module_FULL_TRAINING.png", show_shapes=True)
	else:
		plot_model(model, to_file = "temporal_module.png", show_shapes=True)	

	return model
def record_weights_for_cnn_predict(model, weights_name, subject, flag):



	return model

def sanity_check_image(X, channel, spatial_size):
	# item = X[0,:,:,:]
	item = X[0, :, :, 0]

	item = item.reshape(224, 224, channel)

	cv2.imwrite('sanity_check.png', item)


def gpu_observer():
	''' comment
	nvmlInit()
	for i in range(nvmlDeviceGetCount()):
		handle = nvmlDeviceGetHandleByIndex(i)
		meminfo = nvmlDeviceGetMemoryInfo(handle)
		print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (
			nvmlDeviceGetName(handle),
			meminfo.free/1024.**2, meminfo.used/1024.**2, meminfo.total/1024.**2))    
	'''




