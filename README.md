# face-recognition项目介绍

这个项目有两个大功能：人脸预处理和CNN人脸训练与识别<br />
由于不能上传大文件，需要下载shape_predictor_68_face_landmarks文件，在该文件夹内提供了下载链接<br />
人脸数据需要自己造，暂不提供<br />

'''python
cutting_position = (d.left(), d.top(), d.right(), d.bottom())
# 切割出人脸
im = Image.open(r'C:\Users\zyxrdu\Desktop\w\a4.jpg')
region = im.crop(cutting_position)
# 人脸缩放
a = 500  # 人脸方格大小
if region.size[0] >= a or region.size[1] >= a:
    region.thumbnail((a, a), Image.ANTIALIAS)
else:
    region = region.resize((a, a), Image.ANTIALIAS)
# 将Image转化为cv2
region = cv2.cvtColor(np.asarray(region), cv2.COLOR_RGB2BGR)
# 保存人脸
cv2.imshow('region', region)
print(type(region))
cv2.waitKey(0)
'''

<br />
# 1、人脸预处理<br />
人脸活体检测：从原始图片中识别出人脸的位置，以及68个特征点位置，借助shape_predictor_68_face_landmarks模型。<br />
![image](https://github.com/duhanmin/face-recognition/blob/master/images/4.png)<br /><br />
脸部检测提取：从原始图片中将人脸提取出来<br />
![image](https://github.com/duhanmin/face-recognition/blob/master/images/5.png)<br /><br />
脸部检测提取：人脸旋转，可以将侧脸拉正<br />
![image](https://github.com/duhanmin/face-recognition/blob/master/images/3.png)<br />
<br />
# 2、CNN人脸训练与识别<br />
网络方案：2个卷积、2个pooling、两个全连接、所有激活函数均采用leaky_relu<br />
原始数据截图：<br />
![image](https://github.com/duhanmin/face-recognition/blob/master/images/1.png)<br /><br />
处理后的数据：<br />
![image](https://github.com/duhanmin/face-recognition/blob/master/images/2.png)<br /><br />
训练效果：<br />
![image](https://github.com/duhanmin/face-recognition/blob/master/images/6.png)<br /><br />
测试效果：<br />
![image](https://github.com/duhanmin/face-recognition/blob/master/images/7.png)<br /><br />
