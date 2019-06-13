# -*- coding: utf-8 -*-
import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import pandas as pd

from utilities import Read_Input_Images,  data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import record_scores, LossHistory, filter_objective_samples
def load_db():
	pass


def restructure_data(subject, subperdb, labelpersub, subjects, n_exp, r, w, timesteps_TIM, channel):

	Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(subject, subperdb, labelpersub, subjects, n_exp)

	# Rearrange Training labels into a vector of lack-data, breaking sequence  Train_X  three_dim

	#Train_X three_dim 视频数量（237）X 每个视频图像数量（9）* 每个图像像素数量（224*224*3）

	#转变为空间数据  reshape(,224，224，3）,  train_X_spatial (图像数量，w,r,channel)_
	Train_X_spatial = Train_X.reshape(Train_X.shape[0], channel,r, w)
	Test_X_spatial = Test_X.reshape(Test_X.shape[0], channel,r, w )

	# Extend Y labels 10 fold, so that all lack-data have labels
	Train_Y_spatial = np.repeat(Train_Y, 1, axis=0)
	Test_Y_spatial = np.repeat(Test_Y, 1, axis=0)

	#数据  （图像数量，channel，r,w)
	X = Train_X_spatial.reshape(Train_X_spatial.shape[0], r, w,channel)
	y = Train_Y_spatial.reshape(Train_Y_spatial.shape[0], n_exp)
	normalized_X = X.astype('float32') / 255.

	test_X = Test_X_spatial.reshape(Test_X_spatial.shape[0], r, w,channel)
	test_y = Test_Y_spatial.reshape(Test_Y_spatial.shape[0], n_exp)
	normalized_test_X = test_X.astype('float32') / 255.


	print ("Train_X_shape: " + str(np.shape(Train_X)))
	print ("Train_Y_shape: " + str(np.shape(Train_Y)))
	print ("Test_X_shape: " + str(np.shape(Test_X)))
	print ("Test_Y_shape: " + str(np.shape(Test_Y)))	
	print ("X_shape: " + str(np.shape(X)))
	print ("y_shape: " + str(np.shape(y)))
	print ("test_X_shape: " + str(np.shape(test_X)))	
	print ("test_y_shape: " + str(np.shape(test_y)))

	return Train_X, Train_Y, Test_Y, Test_Y, Test_Y_gt, X, y, test_X, test_y