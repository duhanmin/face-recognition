# face-recognition项目介绍

这个项目有两个大功能：<br />
<br />
1、人脸预处理<br />
人脸活体检测：从原始图片中识别出人脸的位置，以及68个特征点位置，借助shape_predictor_68_face_landmarks模型。<br />
![image](https://github.com/duhanmin/face-recognition/blob/master/images/4.png)
脸部检测提取：从原始图片中将人脸提取出来<br />
脸部检测提取：人脸旋转，可以将侧脸拉正<br />
<br />
2、CNN人脸训练与识别<br />
网络方案：2个卷积、2个pooling、两个全连接、所有激活函数均采用leaky_relu<br />
