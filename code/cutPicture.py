#coding=utf-8
#保存训练数据的moments特征以及对应angle标签
import cv2
import numpy as np
import sys
import time


st = time.time()
moments_num=7
#图像地址
dirTrain='D:\\image\\1\\'
dirResult='D:\\image\\2\\'         
start=3

picNum=153
min_area=15000
f_mean=np.loadtxt("D:\\image\\txt\\2l\\image_train_mean.txt",delimiter=' ')

for iii in range(picNum):

    picIndex=iii+start

    if picIndex!=100:

        picName='00000'

        if picIndex<10:
            picName=picName+'00'+str(picIndex)
        elif picIndex<100:
            picName=picName+'0'+str(picIndex)
        else:
            picName=picName+str(picIndex)

        picNameNew=picName+'.jpg'
        picName=picName+'.TIF'

        print(picName)


        img= cv2.imread(dirTrain+picName, 0)

        
        # 阈值操作
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # 轮廓检测,找到x光片中的检测目标
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, 2)
        l = len(contours)
        f = np.empty([l, moments_num], dtype=float)

        for i in range(l):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area > min_area:
                M = cv2.moments(cnt)
                feature = [M['nu20'], M['nu11'], M['nu02'], M['nu30'], M['nu21'], M['nu12'], M['nu03']]
                f[i, :] = feature

        f = f - f_mean
        f = np.abs(f)
        s = np.sum(f, axis=1)
        index = np.argmin(s)

        cnt = contours[index]
        x, y, w, h = cv2.boundingRect(cnt)
        x = x - 50
        y = y - 50
        w = w + 100
        h = h + 100
        target = img[y:y + h, x:x + w]

        cv2.imwrite(dirResult+picNameNew, target)



        