#coding:utf-8
import sys
sys.path.append(r"/home/deep/mnt_for_mount_creact_by_liuyang/Deep/Weldpool-recognition/")
import argparse  #显示参数信息，字典作用进行参数的传递
from src.cnn_lstm.cnn_lstm_trian import  train_weld
def main(args):  #args通过sparse_args()方法实际上从我们的命令行参数中得到了一些数据
	             # 相当于使用了字典方法进行参数的传递
	dB=[]
	print(args.dB)
	dB.append(args.dB)
	print(dB)
	args.dB=dB  #把dB的列表赋值给args.dB,方便train_weld的参数传入
	print(dB)
	print(args.train)
	if args.train == "./cnn_lstm_train.py":  #cnn_lstm训练
		train_weld(args.batch_size,
				   args.spatial_epochs,
				   args.temporal_epochs,
				   args.train_id,
				   args.dB,
				   args.spatial_size,
				   args.flag,
				   args.timesteps_TIM,
				   args.tensorboard,
				   args.root_db_path,
				   args.model_name)

	else:
		print("请输入正确的引用函数名称。--help ")
# flag list:
	# st -> spatio-temporal 时空
	# s -> spatial only     仅时间
	# t -> temporal only    仅空间
	# nofine - > no finetuning, train svm classifer only  没有微调，只训练svm分类器
	# scratch -> train from scratch  从头开始训练

	# eg for calling more than 1 databases: 例如，用于调用多个数据库：
	# python main.py --dB 'CASME2_Optical' 'CASME2_Strain_TIM10'
	# --batch_size=1 --spatial_epochs=100 --temporal_epochs=100 --train_id='default_test'
	# --spatial_size=224 --flag='st4se'


'''**Example for single db**:
python main.py --dB '0518' --batch_size=32 --spatial_epochs=100
--temporal_epochs=100 --train_id='default_test' --spatial_size=224 --flag='st' --timesteps_TIM=3
--root_db_path '/home/deep/data/Deep/test-data/'
'''
if __name__ == '__main__':
	parser = argparse.ArgumentParser()   #创建解释器对象parser
	# add_argument()方法，用来指定程序需要接受的命令参数
	parser.add_argument('--train', type=str, default='./cnn_lstm_train.py', help='Using which script to train，使用哪个脚本进行训练')
	parser.add_argument('--batch_size', type=int, default=16, help='Training Batch Size，批量训练')
	parser.add_argument('--spatial_epochs', type=int, default=100, help='Epochs to train for Spatial Encoder，空间编码器的训练次数') #10
	parser.add_argument('--temporal_epochs', type= int, default=1000, help='Epochs to train for Temporal Encoder，时间编码器的训练次数') #40
	parser.add_argument('--train_id', type=str, default="1", help='To name the weights of model，命名模型的权重')
	parser.add_argument('--dB', nargs="+", type=str, default='0518', help='Specify Database，指定的数据库')
	parser.add_argument('--spatial_size', type=int, default=224, help='Size of image，图像尺寸')
	parser.add_argument('--flag', type=str, default='st', help='Flags to control type of training，用于控制训练类型的标志')
	parser.add_argument('--timesteps_TIM', type=int, default=20, help='Flags to use either objective class or emotion class，几个图片作为一个动作，序列')
	parser.add_argument('--tensorboard', type=bool, default=True, help='tensorboard display，张量显示')
	parser.add_argument('--root_db_path', type=str, default='../../test-data/', help='data path，数据路径')
	parser.add_argument('--model_name', type=str, default='DenseNet201', help='modelName--> ResNet50,InceptionV3,VGG16,VGG19,Xception,InceptionResNetV2,DenseNet201')
	# default：设置参数的默认值
	# parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")
	# type：把从命令行输入的结果转成设置的类型
	# parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], help="increase output verbosity")
	# help：参数命令的介绍
	''' 
	model_name = "ResNet50"
    model_name = "Xception"
    model_name = "DenseNet201"
    model_name = "InceptionResNetV2"
    model_name = "InceptionV3"
	'''
	args = parser.parse_args()  #parse_args()方法实际上从我们的命令行参数中返回了一些数据  例如'--train'返回参数train=default
	print("__main__args")
	print(args)
	print(type(args))  #argparse.Namespace类型
    #Namespace(batch_size=32, dB='0518', flag='st', model_name='VGG16', root_db_path='../../test-data/', spatial_epochs=1, spatial_size=224, temporal_epochs=10, tensorboard=True, timesteps_TIM=3, train='./train.py', train_id='1')
	#输出结果
	main(args)