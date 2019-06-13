#coding:utf-8
from src.tool.data_prepare import *
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from src.tool.models import *
from src.tool.models import *
from keras.callbacks import EarlyStopping
from src.tool.tool import *
from keras.utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau
import pandas as pd


def train(model,path,train_path,batch_size,nb_classes,spatial_epochs,temporal_epochs,train_id,image_size,flag,timesteps_TIM,tensorboard):
    print(" testing .....       id :"+train_id,
          "model :"+model,
          "path :"+path,
          "dataLabel number:",nb_classes,
          "number of each batch :",batch_size,
          "spatial epochs :",spatial_epochs,
          "temporal_epochs :",temporal_epochs,
          "image size",image_size,
          "useing temporal train? :",flag,
          "size fo lstm :",timesteps_TIM,
          "tensorboard :",tensorboard)

    '''-------------------------------------------------------------------------------------
       -first step: read image from $path ,and trian <<<------>>>lable,                    -
       -and preduce train_datas,train_lables,val_datas,val_lables,test_datas,test_lables   -
       - train : val:test == 6:2:2                                                         -
       -------------------------------------------------------------------------------------
    '''
    #用于预测
    predict_steps=98
    test_step = get_steps_per_epoch(train_path+'test/')

    #用于训练
    epochs = 200
    validation_steps=get_steps_per_epoch(train_path+'validation/')
    # example  all= 22011
    #batch size = 16 batch size 在tool/data_prepare下
    #steps_per_epoch = all / batch size=22011/16=1376
    steps_per_epoch=get_steps_per_epoch(train_path+'train/')
    #数据函数已经成功完成。
   # train_datas, train_lables = get_train_all(train_path='../deal-data/train/',image_size=224,channel=3)
   #val_datas, val_lables = get_val_all(val_path="../deal-data/validation/",image_size=224,channel=3)
   # test_datas, test_lables = get_test_all(test_path='../deal-data/test/',image_size=224,channel=3)
    generate_train = generate_batch_data(train_path+"train")
    generate_validation = generate_batch_data(train_path+"validation")
    generate_test = generate_batch_data(train_path+"test")




    '''-------------------------------------------------------------------------------------
       -second step: biuld a model for yourself,we can choose:                             - 
       - --model: ResNet50,InceptionV3,VGG16,VGG19,Xception,InceptionResNetV2,DenseNet201
       - 就这么跟你说吧，要把模型复制到4个GPU上进行训练，加快速度。
       -------------------------------------------------------------------------------------
    '''
    #model = get_model()
    my_spatial_model = mySpatialModel(model_name=model,
                                      spatial_size=image_size,
                                      nb_classes=2,
                                      channels=3,
                                      weights_path=None)
    multi_gpu_model(my_spatial_model, gpus=4) #多gpu同时进行计算
    '''
       -------------------------------------------------------------------------------------
       - third step: some params must be define                                            - 
       -history: let me known prossese's status                                            -
       -optimizer: let me known which optimizer can I choose.such as sdg adm               -
       -stopping: if accuracy do not improve anymore,just stop prosses.                    -
       -tbCallbacks: for tensorborad      
       -添加学习率到history中，查看学习率的变化。将history中的训练过程数据存文件里面，
       -------------------------------------------------------------------------------------
    '''
    dirname = train_path + 'train_procese/' + train_id + '/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    weight_path=dirname
    log_dir=dirname+'logs'
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    if os.path.exists(weight_path) == False:
        os.makedirs(weight_path)

    history = LossHistory(train_path,model,train_id)
    checkpointer = ModelCheckpoint(filepath=weight_path+model+'_'
                                            +str(train_id)+'_weights.hdf5',
                                   verbose=1, save_best_only=True)
    optimizer = getOptimizer()
    loss_function = getLoss()
    stopping = EarlyStopping(monitor='val_loss',
                             min_delta=0,
                             patience=30,
                             verbose=1,
                             mode='auto',
                             restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=10, min_lr=0.00001,verbose=1)
    tbCallbacks = callbacks.TensorBoard(log_dir,
                                        histogram_freq=1,
                                        write_graph=True,
                                        write_images=True,
                                        write_grads=True)

    '''
       -------------------------------------------------------------------------------------
       - fourth step:                                                                      -
       - start fit ,predict ,evaluate                                                              - 
       - tensorboard                                                                       - 
       -------------------------------------------------------------------------------------
    '''
    my_spatial_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    #my_spatial_model.fit(train_datas, train_lables, batch_size=batch_size, epochs=spatial_epochs, shuffle=True,
    #                     verbose=1,validation_data=(val_datas,val_lables),callbacks=[history, stopping, tbCallbacks])
    '''
    开始训练数据了
    '''
    my_spatial_model.fit_generator(
            generate_train,
            steps_per_epoch=steps_per_epoch, #每次要执行的操作，
            epochs=epochs,
            validation_data = generate_validation,
            validation_steps=validation_steps,
            #callbacks=[history, stopping, tbCallbacks])
            callbacks = [reduce_lr,checkpointer,history]
            # nb_val_samples = len(generate_validation),     #batch_size*batch_size),
            # verbose=1,
    )


    '''
    保存模型权重
    '''
    #my_spatial_model.save_weights("../model_weight_by_liuyang/" +  "classfication_keyhole_or_not" + ".h5")

    '''
     对模型进行预测
    '''
    predict = my_spatial_model.predict_generator(generate_validation,
                                                 steps=validation_steps,
                                                 max_queue_size=50,
                                                 workers=1,
                                                 use_multiprocessing=False,
                                                 verbose=1)
    predict_label = np.argmax(predict, axis=1)
    true_label = generate_validation.classes
    print(predict_label, true_label)
    predict_label = predict_label[0:validation_steps*16]

    table = pd.crosstab(true_label, predict_label, rownames=['label'], colnames=['predict'])
    print("打印预测矩阵")
    print(predict)

    print("打印交叉表")
    print(table)
    '''
     评估模型。
    '''
    loss,accuracy = my_spatial_model.evaluate_generator(generate_test,steps=test_step)
    print("loss: ",loss,"accuracy",accuracy )
    history.loss_plot("epoch")