import tkinter
from tkinter import filedialog,messagebox
#filedialog文件目录
#messagebox是tkinter中的消息框、对话框  https://www.cnblogs.com/progor/p/8506513.html
import cv2
import os
from PIL import Image, ImageTk
import random


######################图片的处理#########################
P='D:/img'
isExists = os.path.exists(P)
if not isExists:
        os.mkdir(P)
Path = P+'/'
top = tkinter.Tk()       # 进入消息循环
top.wm_title("图片处理") #标题
top.geometry("600x600")  #设置窗口大小
top.update()             #刷新窗口
# print("当前窗口的宽度为",top.winfo_width())  #显示大小
# print("当前窗口的高度为",top.winfo_height())
text= tkinter.Label(top, text="请输入剪裁的尺寸")  #Label显示文字组件
text.place(x=10,y=10,anchor='nw')  #place摆放位置显示pack，grid    https://blog.csdn.net/qq_39401420/article/details/80871301
entry =tkinter.Entry(top,width=5) # Entry 单行文本框组件
# Text  多行文本框组件
entry.place(x=10,y=40,anchor='nw')
entry1 =tkinter.Entry(top,width=5)
entry1.place(x=60,y=40,anchor='nw')
def choose():
        global path
        path=tkinter.filedialog.askdirectory()
        print(path)
        label = tkinter.Label(top, text=path)
        label.place(x=10,y=95,anchor='nw')
def cut():
        print(path)
        FileNames = os.listdir(path)
        i = 1
        for file_name in FileNames:
                fullfilename = os.path.join(path, file_name)
                print(fullfilename)
                img = cv2.imread(fullfilename)
                res = cv2.resize(img, (int(entry.get()), int(entry1.get())), interpolation=cv2.INTER_AREA)
                cv2.imwrite(Path + str(i) + '.jpg', res)
                i += 1
        sf=tkinter.messagebox.askyesno("提示",'图片剪裁保存完成，是否要查看')
        if sf==True:
                os.system("start "+P)
        else:
                pass
A = tkinter.Button(top, text ="选择图片路径", command =choose)  #定义按钮组件
A.place(x=10,y=65,anchor='nw')
B=tkinter.Button(top,text="开始剪裁图片并保存"+P+'路径',command=cut)
B.place(x=10,y=120,anchor='nw')
# path='E:/mnist/Deep/test-data/0518/lack-data/jpg1'
# def imgbutton():
#         File = os.listdir(path)
#         for file in File:
#                 fullfilename1 = os.path.join(path, file)
#                 print(fullfilename1)
#                 img_open = Image.open(fullfilename1)
#                 img_jpg = ImageTk.PhotoImage(img_open)
#                 label_img = tkinter.Label(top, image=img_jpg)
#                 label_img.pack()
#                 break
#         # img_open = Image.open('E:/mnist/Deep/test-data/0518/lack-data/jpg1/700.jpg')
#         # img_png = ImageTk.PhotoImage(img_open)
#         # label_img = tkinter.Label(top, image=img_png)
#         # label_img.pack()
# C = tkinter.Button(top, text="下一张", command=imgbutton)
# C.pack()



######################视频的处理#########################
P1='D:/video'
isExists = os.path.exists(P1)
if not isExists:
        os.mkdir(P1)
Path1 = P1+'/'
def choose1():
        global path1
        path1=tkinter.filedialog.askdirectory()
        print(path1)
        label = tkinter.Label(top, text=path1)
        label.place(x=10, y=230)
def cut1():
        print(path1)
        FileNames1 = os.listdir(path1)
        i = 1
        for file_name1 in FileNames1:
                fullfilename1 = os.path.join(path1, file_name1)
                print(fullfilename1)
                # 读取视频文件
                cap = cv2.VideoCapture(fullfilename1)
                # 获取视频文件的帧率
                print(cap.get(5))
                success, frame = cap.read()
                # count：记录视频帧
                count = 1
                while success:
                        # 把当前视频帧存放到对应文件夹下
                        cv2.imwrite(Path1+str(i)+'_'+str(count)+".jpg",frame)  # save frame as JPEG file
                        success, frame = cap.read()
                        count += 1
                i+=1
        cv2.destroyAllWindows()
        cap.release()
        sf1=tkinter.messagebox.askyesno('提示','视频帧保存完成,是否要查看')
        if sf1==True:
                os.system("start "+P1)
        else:
                pass
D = tkinter.Button(top, text ="选择视频路径", command =choose1)
D.place(x=10,y=200)
E = tkinter.Button(top, text ="开始保存视频帧在"+P1+"路径", command =cut1)
E.place(x=10,y=255)


top.mainloop()



