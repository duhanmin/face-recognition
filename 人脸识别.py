#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
#@version: python3
#@author: duhanmin
#@contact: duhanmin@foxmail.com
#@software: PyCharm Community Edition
#@file: 人脸识别.py
#@time: 2017/12/6 16:32
'''

from skimage import io, transform
import os
import tensorflow as tf
import numpy as np
from 人脸标准化 import normalization
import sklearn.preprocessing

# 数据集地址
path = 'C:/Users/zyxrdu/Desktop/cnn_xunlian/'
# 处理后的数据集
path_normalization = 'C:/Users/zyxrdu/Desktop/cnn_xunlian/normalization/'
# 模型保存地址
model_path = '../人脸识别/model.ckpt'
# tfrecord文件存放路径
TFRECORD_FILE = "../人脸识别/tfrecords/"

#全局one-hot编码空间
label_binarizer = ""

#默认按列归一化
def def_normalization(x,h = 1):
    # 记录归一化全局最大值最小值，回代时需要
    if h==1:
        #按列归一化
        amin, amax = np.min(x, 0), np.max(x, 0)
        xx = (x - amin) / (amax - amin)
    else:
        #按行归一化
        amin, amax = np.min(x, 1), np.max(x, 1)
        xx = ((x.T - amin) / (amax - amin)).T
    #记录归一化最大值最小值，回代时需要
    return xx

#使用one-hot编码，将离散特征的取值扩展到了欧式空间
def def_one_hot(x):
    if label_binarizer == "":
        binarizer = sklearn.preprocessing.LabelBinarizer()
    else:
        binarizer = label_binarizer
    binarizer.fit(range(max(x)+1))
    y= binarizer.transform(x)
    return y


def read_img(path):
    map_path, map_relative = [path + x for x in os.listdir(path) if os.path.isfile(path + x)], [y for y in os.listdir(path)]
    return map_path, map_relative

# 读取图片并处理保存
def read_dispose_img(path):
    map_path ,map_relative= read_img(path)
    imgs = []
    labels = []
    for idx, folder, in enumerate(map_path):
        print("读取图并处理中......"+path_normalization+map_relative[idx])
        normalization(folder,path_normalization+map_relative[idx])

# 读取处理后的图片，顺序打乱，划分测试和训练
def read_new_img(path):
    map_path, map_relative = read_img(path)
    imgs=[]
    labels=[]
    for idx, folder, in enumerate(map_path):
        img = io.imread(path_normalization+map_relative[idx])
        img = transform.resize(img, (50, 50))
        imgs.append(img)
        labels.append(int(map_relative[idx].split(",")[0].split("_")[0]))
    x_data, x_label = np.array(imgs), np.array(labels)

    # # 打乱顺序
    # num_example = data.shape[0]
    # arr = np.arange(num_example)
    # np.random.shuffle(arr)
    # data = data[arr]
    # label = label[arr]
    # # 将所有数据分为训练集和验证集
    # ratio = 0.8
    # s = np.int(num_example * ratio)
    # return data[:s], data[s:], def_one_hot(label[:s]), def_one_hot(label[s:])
    map_path, map_relative = read_img(path + 'c/')
    imgs=[]
    labels=[]
    for idx, folder, in enumerate(map_path):
        img = io.imread(path_normalization + 'c/' +map_relative[idx])
        img = transform.resize(img, (50, 50))
        imgs.append(img)
        labels.append(int(map_relative[idx].split(",")[0].split("_")[0]))
    c_data, c_label = np.array(imgs), np.array(labels)
    x_data, c_data, def_one_hot(x_label), def_one_hot(c_label)
    return x_data, c_data, def_one_hot(x_label), def_one_hot(c_label)


#初始化权值
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.01)#生成一个截断的正态分布
    return tf.Variable(initial,name=name)

#初始化偏置
def bias_variable(shape,name):
    initial = tf.constant(0.01,shape=shape)
    return tf.Variable(initial,name=name)

#卷积层
def conv2d(x,W):
    #x input tensor of shape `[batch, in_height, in_width, in_channels]`
    #W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #`strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    #padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def xunlianlo(path):
    # read_dispose_img(path)
    x_train, x_test, y_train, y_test = read_new_img(path_normalization)
    m,n = y_train.shape
    # 命名空间
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 50, 50, 3], name='x-input')
    y = tf.placeholder(tf.float32, [None, n], name='y-input')
    # 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
    x_image = tf.reshape(x, [-1, 50, 50, 3], name='x_image')

    # 初始化第一个卷积层的权值和偏置
    W_conv1 = weight_variable([5, 5, 3, 32], name='W_conv1')  # 5*5的采样窗口，32个卷积核从3个平面抽取特征
    b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    h_conv1 = tf.nn.leaky_relu(conv2d_1)
    h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling

    # 初始化第二个卷积层的权值和偏置
    W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值

    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数

    conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_conv2 = tf.nn.leaky_relu(conv2d_2)
    h_pool2 = max_pool_2x2(h_conv2)  # 进行max-pooling

    # 300*300的图片第一次卷积后还是300*300，第一次池化后变为150*150
    # 第二次卷积后为150*150，第二次池化后变为了75*75
    # 进过上面操作后得到64张7*7的平面

    # 初始化第一个全连接层的权值
    W_fc1 = weight_variable([13 * 13 * 64, 1024], name='W_fc1')  # 上一场有75*75*64个神经元，全连接层有1024个神经元
    b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点

    # 把池化层2的输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2, [-1, 13 * 13 * 64], name='h_pool2_flat')

    # 求第一个全连接层的输出
    wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1 = tf.nn.leaky_relu(wx_plus_b1)

    # keep_prob用来表示神经元的输出概率
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    # 初始化第二个全连接层
    W_fc2 = weight_variable([1024, n], name='W_fc2')
    b_fc2 = bias_variable([n], name='b_fc2')
    wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # 计算输出
    prediction = tf.nn.leaky_relu(wx_plus_b2)

    tf.add_to_collection('predictions', prediction)

    p = tf.nn.softmax(wx_plus_b2)
    tf.add_to_collection('p', p)
    # 交叉熵代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')

    # 使用AdamOptimizer进行优化
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(5).minimize(cross_entropy)

    # 求准确率
    # 结果存放在一个布尔列表中
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #保存模型使用环境
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner, 此时文件名队列已经进队
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(100001):
            # 训练模型
            sess.run(train_step, feed_dict={x: x_train, y: y_train, keep_prob: 0.9})

            test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: x_train, y: y_train, keep_prob: 1.0})
            print("训练第 " + str(i) + " 次, 训练集准确率= " + str(train_acc) + " , 测试集准确率= " + str(test_acc))

            if test_acc == 1 and train_acc >= 0.95:
                print("准确率完爆了")
                # 保存模型
                saver.save(sess, 'nn/my_net.ckpt')
                break

        # 通知其他线程关闭
        coord.request_stop()
        # 其他所有线程关闭之后，这一函数才能返回
        coord.join(threads)

def test_main():
    # 本地情况下生成数据
    x_train, x_test, y_train, y_test = read_new_img(path_normalization)

    m,n = y_test.shape

    # 迭代网络
    with tf.Session() as sess:
        # 保存模型使用环境
        saver = tf.train.import_meta_graph("nn/my_net.ckpt.meta")
        saver.restore(sess, 'nn/my_net.ckpt')

        predictions = tf.get_collection('predictions')[0]
        p = tf.get_collection('p')[0]

        graph = tf.get_default_graph()

        input_x = graph.get_operation_by_name('x-input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        for i in range(m):
            result = sess.run(predictions, feed_dict={input_x: np.array([x_test[i]]),keep_prob:1.0})
            haha = sess.run(p, feed_dict={input_x: np.array([x_test[i]]), keep_prob: 1.0})
            print(haha)
            print("实际 :"+str(np.argmax(y_test[i]))+" ,预测: "+str(np.argmax(result))+" ,预测可靠度: "+str(np.max(haha)))


test_main()
# xunlianlo(path_normalization)
# if __name__ == '__main__':
    #训练模型
    # xunlianlo(path_normalization)
    #测试模型
    # test_main()
    # 处理照片
    # read_dispose_img(path)
