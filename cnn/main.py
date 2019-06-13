#coding:utf-8
import sys
sys.path.append(r"./")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import argparse
from src.cnn.cnn_train  import *
from src.cnn.cnn_test import *
def main(args):
    if args.train == "./cnn_train.py":
        train(args.model,
              args.path,
              args.train_path,
              args.nb_classes,
              args.batch_size,
              args.spatial_epochs,
              args.temporal_epochs,
              args.train_id,
              args.image_size,
              args.flag,
              args.timesteps_TIM,
              args.tensorboard)
    else:
        test(args.model,
              args.path,
              args.nb_classes,
              args.batch_size,
              args.spatial_epochs,
              args.temporal_epochs,
              args.train_id,
              args.image_size,
              args.flag,
              args.timesteps_TIM,
              args.tensorboard)
        print("pelase resign this progress in main.py and fllow")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建解释器对象parser
    # add_argument()方法，用来指定程序需要接受的命令参数
    parser.add_argument('--train', type=str, default='./cnn_train.py',help='Using which script to train，使用哪个脚本进行训练')
    parser.add_argument('--model', type=str, default='InceptionResNetV2',help='InceptionV3--> ResNet50,InceptionV3,VGG16,VGG19,Xception,InceptionResNetV2,DenseNet201')
    parser.add_argument('--path', type=str, default='../../test-data/',help='data path，数据路径')

    parser.add_argument('--train_path', type=str, default='D:/DeepTool/DeepTool/deal-data-whole2/', help='')

    parser.add_argument('--nb_classes', type=int, default=2,help='dataLabel numbers,希望将数据分为个类别')
    parser.add_argument('--batch_size', type=int, default=32,help='Training Batch Size，批量训练')
    parser.add_argument('--spatial_epochs', type=int, default=50,help='Epochs to train for Spatial Encoder，空间编码器的训练次数')  # 10
    parser.add_argument('--temporal_epochs', type=int, default=10,help='Epochs to train for Temporal Encoder，时间编码器的训练次数')  # 40
    parser.add_argument('--train_id', type=str, default="11",help='To name the weights of model，命名模型的权重')
    parser.add_argument('--dB', nargs="+", type=str, default='0518',help='Specify Database，指定的数据库')
    parser.add_argument('--image_size', type=int, default=224,help='Size of image，图像尺寸')
    parser.add_argument('--flag', type=str, default='1',help='Flags to control type of training，用于控制训练类型的标志')
    parser.add_argument('--timesteps_TIM', type=int, default=3,help='Flags to use either objective class or emotion class，几个图片作为一个动作，序列')
    parser.add_argument('--tensorboard', type=bool, default=True,help='tensorboard display，张量显示')
    args = parser.parse_args()  #parse_args()方法实际上从我们的命令行参数中返回了一些数据  例如'--train'返回参数train=default
    main(args)
