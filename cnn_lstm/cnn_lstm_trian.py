# coding=UTF-8
import numpy as np
import glob, os
import gc

from sklearn.metrics import confusion_matrix

from keras import metrics

from keras import optimizers
from keras.models import load_model
import keras
from keras.callbacks import EarlyStopping


from src.cnn_lstm.evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from src.cnn_lstm.utilities import  Read_Batch_Input_Images, data_loader_with_LOSO, label_matching
from src.cnn_lstm.utilities import record_loss_accuracy, record_weights, record_scores, LossHistory  # recording scripts
from src.cnn_lstm.utilities import sanity_check_image, gpu_observer

from src.cnn_lstm.list_databases import restructure_data
from src.cnn_lstm.models import  temporal_module,mySpatialModel,mySpatialModelChannelTest
from keras import backend as K
'''
todo:2019.02.02
1.loss理解与编码:loss
2.编写CNN训练模型
3.ResNet50模型深入理解，从卷积到激活
4.交叉验证
5.tensorboard 显示plot
6.plot model image
'''
def train_weld(batch_size, spatial_epochs, temporal_epochs, train_id, list_dB, spatial_size, flag,
          timesteps_TIM, tensorboard, root_db_path,model_name='ResNet50'):

    ############## Path Preparation ######################
    tensorboard_path = root_db_path + "tensorboard/"
    #stopping = EarlyStopping(monitor='loss', min_delta=0, mode='min')
    if os.path.isdir(root_db_path + 'weights/' + str(train_id)) == False:
        os.mkdir(root_db_path + 'weights/' + str(train_id))

    ############## Variables ###################
    print("Init variable......")
    r = w = spatial_size
    subjects = 26
    samples = 246
    n_exp = 2
    samples = 8000
    data_dim = r * w
    channel = 3
    dB = list_dB[0]
    db_home = root_db_path + dB + "/"
    test_count = 18  # test sample测试样本数目 18个实验

    # total confusion matrix to be used in the computation of f1 score
    tot_mat = np.zeros((n_exp, n_exp))
    history = LossHistory()
    stopping = EarlyStopping(monitor='loss', min_delta=0, mode='min')

    ############## Flags ####################
    tensorboard_flag = tensorboard
    resizedFlag = 1
    train_spatial_flag = 0
    train_temporal_flag = 0
    finetuning_flag = 0
    channel = 3
    serial_count=0 #
    if flag == 'st':
        train_spatial_flag = 1
        train_temporal_flag = 1
        finetuning_flag = 1

    K.set_image_dim_ordering('th')
    sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.00001, decay=0.000001)
    # Different Conditions for Temporal Learning ONLY
    data_dim =1024

    print("Beginning training process.")
    ########### Training Process ############

    labelperSub = label_matching(db_home + "dataLabel", test_count)
    leni = len(labelperSub[4])

    print("Loaded Labels into the tray.")



    ############### Reinitialization & weights reset of models ########################
    print("initialization & weights reset of models .")
    temporal_model = temporal_module(data_dim=data_dim, timesteps_TIM=timesteps_TIM, classes=n_exp)
    temporal_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])




    '''
    my_spatial_model=mySpatialModel(model_name,spatial_size=spatial_size, nb_classes=2, channels=3,
                                                         channel_first=True, weights_path=None,
				   lr=0.005, decay=1e-6, momentum=0.9,plot_model=True)

    '''
    spatial_weights= '../../model_weight/'+model_name+'_'+str(train_id)+'_'+'weights.hdf5'
    my_spatial_model = load_model(spatial_weights)
    spatial_model_feature = record_weights(my_spatial_model,flag)
    '''
    mySpatialModelChannelTest(model_name, spatial_size=spatial_size, nb_classes=2,
                                                            channels=3, channel_first=True, weights_path=None,
                                                            lr=0.005, decay=1e-6, momentum=0.9, plot_model=True)
    '''
    #spatial_model_feature=None

    for sub in range(1, test_count + 1):
        print("starting one test sample--Test>" + str(sub) + "-->")

        print("Reading Images......")
        db_images = db_home + "lack-data/"
        SubperdB = Read_Batch_Input_Images(sub, db_images, resizedFlag, spatial_size, channel)
        lenk = len(SubperdB)
        print("Loaded Images into the tray!")

        gpu_observer()
        weights_path = root_db_path + 'weights/' + str(train_id)
        if (os.path.exists(weights_path) == False):
            os.mkdir(weights_path)

        temporal_weights_name = weights_path + '/temporal_ID_' + str(train_id) + '_' + str(dB) + '_'

        ############ for tensorboard ###############
        if tensorboard_flag == 1:
            cat_path = tensorboard_path + str(sub) + "/"
            if (os.path.exists(cat_path) == False):
                os.mkdir(cat_path)
            tbCallBack = keras.callbacks.TensorBoard(log_dir=cat_path, write_graph=True)
            cat_path2 = tensorboard_path + str(sub) + "spatial/"
            if (os.path.exists(cat_path2) == False):
                os.mkdir(cat_path2)
            tbCallBack2 = keras.callbacks.TensorBoard(log_dir=cat_path2, write_graph=True)
        #############################################


        Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt, X, y, test_X, test_y = \
            restructure_data(sub - 1, SubperdB, labelperSub, subjects, n_exp, r, w, timesteps_TIM, channel)
        gpu_observer()

        print("Beginning training & testing.")
        ##################### Training & Testing #########################
        if train_spatial_flag == 1 and train_temporal_flag == 1:

            print("Beginning spatial training.")
            # Spatial Training
            '''
            my_spatial_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs,shuffle=True,callbacks=[history, stopping, tbCallBack2])
            '''
            print(".record f1 and loss")
            #record_loss_accuracy(db_home, train_id, dB, history) # record f1 and loss

            print(".save  weights")

            print(".spatial encoding")
            # Spatial Encoding
            output = spatial_model_feature.predict(X, batch_size=batch_size)
            #转化为时间序列 ，首先取整。
            output = output[0:int(output.shape[0] / timesteps_TIM) * timesteps_TIM:1]
            serial_count = int(output.shape[0] / timesteps_TIM)
            #feature output.shape[0]图像的数量；serial_count 根据步长timesteps_TIM得出的序列，
            # 一个序列有timesteps_TIM幅图像。output.shape[1]每个图像的特征
            features = output.reshape(serial_count, timesteps_TIM, output.shape[1])

            Train_Y = Train_Y[0::timesteps_TIM]
            Train_Y = Train_Y[0:serial_count]

            print("Beginning temporal training.")
            #刘洋增加代码，金奖features的shape从新弄一下，以适应LSTM的输入要求。
            features =  features[:,:,0:1024]
            # Temporal Training
            print("features's shape:",features.shape,"Train_y's shape: ",Train_Y.shape)
            if tensorboard_flag == 1:
                temporal_model.fit(features, Train_Y, batch_size=batch_size, epochs=temporal_epochs,
                                   callbacks=[tbCallBack]) #这里的[]中可以加入stopping
            else:
                temporal_model.fit(features, Train_Y, batch_size=batch_size, epochs=temporal_epochs)

            print(".save temportal weights")
            # save temporal weights
            #temporal_model = record_weights(temporal_model, temporal_weights_name, sub, 't')  # let the flag be t

            print("Beginning testing.")
            print(".predicting with spatial model")
            # Testing
            output = spatial_model_feature.predict(test_X, batch_size=batch_size)

            print(".outputing features")

            output = output[0:int(output.shape[0] / timesteps_TIM) * timesteps_TIM:1]
            serial_count = int(output.shape[0] / timesteps_TIM)
            features = output.reshape(serial_count, timesteps_TIM, output.shape[1])
            #，为了能让Inception 和Lstm正常结合起来。
            features = features[:,:,0:1024]
            print(".predicting with temporal model")
            predict = temporal_model.predict_classes(features, batch_size=batch_size)
        ##############################################################
            loss,accuracy = temporal_model.evaluate(features, Train_Y,verbose=0)
            print("loss:",loss,"accuracy:",accuracy)

        #################### Confusion Matrix Construction #############
        print (predict)

        Test_Y_gt = Test_Y_gt[0::timesteps_TIM]
        Test_Y_gt = Test_Y_gt[0:serial_count]
        print (np.array(Test_Y_gt, dtype=np.int))

        print(".writing predicts to file")
        file = open(db_home + 'result/predicts_' + str(train_id) + 'lstm_by_liuyang.txt', 'a')
        file.write("predicts_sub_" + str(sub) + "," + (",".join(repr(e) for e in predict.astype(list))) + "\n")
        file.write("actuals_sub_" + str(sub) + "," + (
        ",".join(repr(e) for e in np.array(Test_Y_gt, dtype=np.int).astype(list))) + "\n")
        file.close()

        ct = confusion_matrix(np.array(Test_Y_gt, dtype=np.int), predict)
        # check the order of the CT
        order = np.unique(np.concatenate((predict, Test_Y_gt)))

        # create an array to hold the CT for each CV
        mat = np.zeros((n_exp, n_exp))
        # put the order accordingly, in order to form the overall ConfusionMat
        for m in range(len(order)):
            for n in range(len(order)):
                mat[int(order[m]), int(order[n])] = ct[m, n]

        tot_mat = mat + tot_mat
        ################################################################

        #################### cumulative f1 plotting ######################
        microAcc = np.trace(tot_mat) / np.sum(tot_mat)
        [f1, precision, recall] = fpr(tot_mat, n_exp)

        file = open(db_home + 'result/' + 'f1_' + str(train_id) + 'lstm_by_liuyang.txt', 'a')
        file.write(str(f1) + "\n")
        file.close()
        ##################################################################

        ################# write each CT of each CV into .txt file #####################
        record_scores(db_home, dB, ct, sub, order, tot_mat, n_exp, subjects)
        #下面一行是刘阳加的
#        score = temporal_model.evaluate(test_X, test_y, verbose=0)
        war = weighted_average_recall(tot_mat, n_exp, samples)
        uar = unweighted_average_recall(tot_mat, n_exp)
        print("war: " + str(war))
        print("uar: " + str(uar))
        # code by liuyang ,将loss和accuracy以及war写入文件
        file = open(db_home + 'result/' + 'last_word' + str(train_id) + 'lstm_by_liuyang.txt', 'a')
        # db_home='../../test-data/0518/result/fi_1.txt'
        file.write(str(str(war) + " " + str(uar) + "\n"))  # 写入fi数据
        file.close()
        ###############################################################################

        ################## free memory ####################


        del Train_X, Test_X, X, y

        gc.collect()
    ###################################################

    del my_spatial_model
    del temporal_model
    del spatial_model_feature
