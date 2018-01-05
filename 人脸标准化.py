#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
#@version: python3
#@author: duhanmin
#@contact: duhanmin@foxmail.com
#@software: PyCharm Community Edition
#@file: 人脸标准化.py
#@time: 2017/12/6 16:56
'''

import face_recognition
from PIL import Image
import cv2
import dlib
import numpy as np

#通过三个点计算夹角
#b为夹角位置所在的点
#默认θ=1计算弧度，θ!不等于1时计算角度
def cos_angle(a,b,c,θ = 1):
    x,y = b-a,b-c
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y)/(Lx*Ly)
    # 根据条件选择是计算弧度还是角度
    if θ != 1:
        return np.arccos(cos_angle)*360/2/np.pi
    else:
        return np.arccos(cos_angle)

def trait_angle(path):
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(
        r'shape_predictor_68_face_landmarks/shape_predictor_68_face_landmarks.dat')
    img = cv2.imread(path)
    faces = detector(img, 1)
    feas = []  # 关键点
    if (len(faces) > 0):
        for k, d in enumerate(faces):
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
            shape = landmark_predictor(img, d)

            for i in range(68):
                num = str(shape.part(i))[1:-1].split(",")
                feas.append([int(num[0]), int(num[1])])

    feas = np.array(feas)
    s_fa = feas[45, :][1] - feas[36, :][1]
    a, b, c = feas[45, :], feas[36, :], np.array(feas[45, :][0], feas[36, :][1])
    if abs(s_fa) > 5:
        if s_fa > 0 and cos_angle(a, b, c,θ=4) >35:
            return cos_angle(a, b, c,θ=4)
        elif s_fa < 0 and cos_angle(a, b, c,θ=4) >35:
            return 360-cos_angle(a, b, c, θ=4)
        else:
            return 0
    else:
        return 0

def normalization(input,output):
    path =input
    out_path = output

    # 读取图片并识别人脸
    img = face_recognition.load_image_file(path)
    face_locations = tuple(list(face_recognition.face_locations(img)[0]))

    # 重新确定切割位置并切割
    top = face_locations[0]
    right = face_locations[1]
    bottom = face_locations[2]
    left = face_locations[3]
    cutting_position = (left, top, right, bottom)
    # 切割出人脸
    im = Image.open(path)

    region = im.crop(cutting_position)

    # 人脸缩放
    a = 50  # 人脸方格大小
    if region.size[0] >= a or region.size[1] >= a:
        region.thumbnail((a, a), Image.ANTIALIAS)
    else:
        region = region.resize((a, a), Image.ANTIALIAS)
    # 人脸旋转
    θ =trait_angle(path)
    # region = region.rotate(θ)
    # 保存人脸
    region.save(out_path)
