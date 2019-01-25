#coding=utf-8
#保存训练数据的moments特征以及对应angle标签
import cv2
import numpy as np
import sys
import time


st = time.time()

#图像地址
dirTrain='D:\\image\\new1\\'
		   




picNum=152
#中心矩数
moments_num=7

angle_num=2


labels=np.empty([picNum,angle_num],dtype=float)


features=np.empty([picNum,moments_num],dtype=float)



flag=0
#读取图像名字txt文件
image_train_f=open(dirTrain+'image_train.txt','r')
img_name_train=image_train_f.readline()		
img_name_train=img_name_train.strip('\n')	
image_train_list=open(dirTrain+'image_train_list.txt','w')

while img_name_train:

	print(flag)

	img = cv2.imread(dirTrain+img_name_train,0)
	ret,thresh = cv2.threshold(img,50,255,0)
	
	_,contours,hierarchy = cv2.findContours(thresh, 1, 2)
	cnt = contours[0]
	M = cv2.moments(cnt)
	features[flag]=[M['nu20'],M['nu11'],M['nu02'],M['nu30'],M['nu21'],M['nu12'],M['nu03']]

	if np.sum(np.abs(features[flag]))>0.2:

		image_train_list.write(img_name_train+'\n')

		xyz_train=(img_name_train.strip('.jpg')).split('_')
		x_train=float(xyz_train[0])
		y_train=float(xyz_train[1])
		labels[flag]=[x_train,y_train]

		flag=flag+1
	img_name_train=image_train_f.readline()		
	img_name_train=img_name_train.strip('\n')

f_mean=np.mean(features, axis=0)

image_train_f.close()
image_train_list.close()


featuress=np.empty([flag,moments_num],dtype=float)
labelss=np.empty([flag,angle_num],dtype=float)



for i in range(flag):
	featuress[i]=features[i]
	labelss[i]=labels[i]


np.savetxt(dirTrain+'image_train_features.txt',featuress,fmt='%f')
np.savetxt(dirTrain+'image_train_labels.txt',labelss,fmt='%f')

np.savetxt(dirTrain+'image_train_mean.txt',f_mean,fmt='%f')

print('time: {:.3f}.'.format(time.time()-st))
print('done')