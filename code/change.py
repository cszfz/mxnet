import cv2


fs=[0,1,2,4,6,8,10,-2,-4,-6,-8,-10]
xs=[11,11,6,6,6,6,6,6,6,6,6,6]

starts=[3,24,45,56,67,78,89,101,112,123,134,145]

dirFile='D:\\image\\2\\'

for i in range(12):
	
	
	f=fs[i]

	x=xs[i]

	start=starts[i]

	for j in range(x):

		picName='00000'

		picNum=start+j

		if picNum <10:
			picName=picName+'00'+str(picNum)
		elif picNum<100:
			picName=picName+'0'+str(picNum)

		else :
			picName=picName+str(picNum)

		picPath=dirFile+picName+'.jpg'
		#print(picPath)

		img=cv2.imread(picPath)
		

		newPicname='D:\\image\\new1\\'+str(f)+'_'+str(-j)+'.jpg'

		#print(newPicname)
		print(str(f)+'_'+str(-j)+'.jpg')

		cv2.imwrite(newPicname,img)
	

