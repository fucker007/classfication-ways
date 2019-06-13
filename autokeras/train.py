import sys
sys.path.append("/home/deep/PycharmProjects/autokeras")
from autokeras.image.image_supervised import load_image_dataset
import numpy as np
x_train_keyhole, y_train_keyhole = load_image_dataset(csv_file_path="deal-data/train/train_keyhole.csv",
                                      images_path="deal-data/train/the_keyhole")
print(x_train_keyhole.shape)
print(y_train_keyhole.shape)

x_train_no_keyhole, y_train_no_keyhole = load_image_dataset(csv_file_path="deal-data/train/train_no_keyhole.csv",
                                      images_path="deal-data/train/no_keyhole")
print(x_train_no_keyhole.shape)
print(y_train_no_keyhole.shape)
#将训练数据加起来
x_train,y_train = np.vstack([x_train_keyhole,x_train_no_keyhole]),np.hstack([y_train_keyhole,y_train_no_keyhole ])
print("x_train:",x_train.shape)
print("y_train",y_train.shape)

x_test_keyhole, y_test_keyhole = load_image_dataset(csv_file_path="deal-data/test/test_keyhole.csv",
                                    images_path="deal-data/test/the_keyhole")
print("the key hole:",x_test_keyhole.shape)
print(y_test_keyhole.shape)

x_test_no_keyhole, y_test_no_keyhole = load_image_dataset(csv_file_path="deal-data/test/test_no_keyhole.csv",
                                    images_path="deal-data/test/no_keyhole")
print(x_test_no_keyhole.shape)
print(y_test_no_keyhole.shape)
x_test,y_test = np.vstack((x_test_keyhole,x_test_no_keyhole)),np.hstack((y_test_keyhole,y_test_no_keyhole ))

print("x_test:",x_test.shape)
print("y_test:",y_test.shape)
from keras.datasets import mnist
from autokeras.image.image_supervised import ImageClassifier

if __name__ == '__main__':

    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_train = x_train.reshape(x_train.shape + (1,))
    #x_test = x_test.reshape(x_test.shape + (1,))
    model_file_name =  "./deal-data/model/autokeras.h5"
    clf = ImageClassifier(path="./deal-data/show_net",verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    clf.export_autokeras_model(model_file_name)
    y = clf.evaluate(x_test, y_test)
    print(y)
