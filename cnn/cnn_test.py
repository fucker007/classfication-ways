# coding:utf-8
from src.tool.data_prepare import *
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from src.tool.models import *
from src.tool.models import *
from keras.callbacks import EarlyStopping
from src.tool.tool import *
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
import pandas as pd


def test(model, path, batch_size, nb_classes, spatial_epochs, temporal_epochs, train_id, image_size, flag,
          timesteps_TIM, tensorboard):
    print("id :" + train_id,
          "model :" + model,
          "path :" + path,
          "dataLabel number:", nb_classes,
          "number of each batch :", batch_size,
          "spatial epochs :", spatial_epochs,
          "temporal_epochs :", temporal_epochs,
          "image size", image_size,
          "useing temporal train? :", flag,
          "size fo lstm :", timesteps_TIM,
          "tensorboard :", tensorboard)

    '''-------------------------------------------------------------------------------------
       -first step: read image from $path ,and trian <<<------>>>lable,                    -
       -and preduce train_datas,train_lables,val_datas,val_lables,test_datas,test_lables   -
       - train : val:test == 6:2:2                                                         -
       -------------------------------------------------------------------------------------
    '''
    # 数据函数已经成功完成
    generate_train = generate_batch_data("../../deal-data/train")
    generate_validation = generate_batch_data("../../deal-data/validation")
    generate_test = generate_batch_data("../../deal-data/test")

    '''-------------------------------------------------------------------------------------
       -second step: biuld a model for yourself,we can choose:                             - 
       - --model: ResNet50,InceptionV3,VGG16,VGG19,Xception,InceptionResNetV2,DenseNet201
       - 就这么跟你说吧，要把模型复制到4个GPU上进行训练，加快速度。
       -------------------------------------------------------------------------------------
    '''

    my_spatial_model = load_model('../../model_weight/'+model+'_'
                                            +str(train_id)+'_weights.hdf5')  # 加载权重到模型中

    '''
     对模型进行预测
    '''
    predict = my_spatial_model.predict_generator(generate_validation, steps=98, max_queue_size=50, workers=1,
                                                 use_multiprocessing=False, verbose=1)
    predict_label = np.argmax(predict, axis=1)
    true_label = generate_validation.classes
    print(predict_label, true_label)
    predict_label = predict_label[0:1548]
    table = pd.crosstab(true_label, predict_label, rownames=['label'], colnames=['predict'])
    print("打印预测矩阵")
    print(predict)
    print("打印交叉表")
    print(table)
    '''
     评估模型。
    '''
    loss, accuracy = my_spatial_model.evaluate_generator(generate_test, steps=98)
    print("loss: ", loss, "accuracy", accuracy)
