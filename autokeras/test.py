from keras.utils import plot_model
from keras.models import load_model
from keras.models import Sequential
from autokeras.utils import pickle_from_file
import sys
import numpy as np
sys.path.append("/home/deep/PycharmProjects/autokeras")
from autokeras.image.image_supervised import load_image_dataset

x_test_keyhole, y_test_keyhole = load_image_dataset(csv_file_path="deal-data/test/test_keyhole.csv",
                                    images_path="deal-data/test/the_keyhole")
print("the key hole:",x_test_keyhole.shape)
print(y_test_keyhole.shape)

x_test_no_keyhole, y_test_no_keyhole = load_image_dataset(csv_file_path="deal-data/test/test_no_keyhole.csv",
                                    images_path="deal-data/test/no_keyhole")
print(x_test_no_keyhole.shape)
print(y_test_no_keyhole.shape)
x_test,y_test = np.vstack((x_test_keyhole,x_test_no_keyhole)),np.hstack((y_test_keyhole,y_test_no_keyhole ))

model_file_name = 'deal-data/model/autokeras.h5' #加载模型
#model = load_model('deal-data/model/autokeras.h5')
model = pickle_from_file(model_file_name)
#model = Sequential(model)
#plot_model(model, to_file='my_model.png')
results = model.evaluate(x_test,y_test) #用测试数据测试
print(results)#打印结果