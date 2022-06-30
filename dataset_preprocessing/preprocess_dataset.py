#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:56:43 2019

@author: tpetit
"""

import argparse
import numpy as np
import cv2, dlib
import os
from tqdm import tqdm

cwd = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('datapath', type=str, help = 'Path to dataset.')
parser.add_argument('resultpath', type=str, help = 'Name of the file to write results.')
parser.add_argument('-m', '--modelpath', type=str, default = 'opencv_face_detector_uint8.pb', help = 'Name of the model to load.')
parser.add_argument('-c', '--configpath', type=str, default = 'opencv_face_detector.pbtxt', help = 'Name of the config file.')
parser.add_argument('-s', '--shapepredictor', type=str, default = 'shape_predictor_68_face_landmarks.dat', help = 'Name of the shape predictor model.')

args = parser.parse_args()

datapath = args.datapath
resultpath = args.resultpath
modelFile = args.modelpath
configFile = args.configpath
net = cv2.dnn.readNetFromTensorflow(os.path.join(cwd, modelFile), os.path.join(cwd, configFile))
predictor = dlib.shape_predictor(os.path.join(cwd, args.shapepredictor))

noise = False
padding = True

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]


keys = os.listdir(datapath)
for k in tqdm(keys) :
    imgs = os.listdir(os.path.join(datapath, k))
    if not os.path.exists(os.path.join(resultpath, k)):
        os.makedirs(os.path.join(resultpath, k))
    for i, img in enumerate(imgs) :
        X = cv2.imread(os.path.join(datapath, k, img))
        if noise :
            X = X/255
            n = np.random.normal(0, 0.01, X.shape)
            X += n
            X = np.fmax(np.fmin(X*255, 255*np.ones(X.shape)), np.zeros(X.shape)).astype('uint8')
        if padding :
            X = cv2.copyMakeBorder(X, int(X.shape[1]*0.5), 0, 0, 0, cv2.BORDER_CONSTANT)
        if X is None :
            print(os.path.join(datapath, k, img))
        frameHeight = X.shape[0]
        frameWidth = X.shape[1]
        blob = cv2.dnn.blobFromImage(X, 1.0, (300, 300), [104, 117, 123], True, False)
        
        # Apply face detection
        net.setInput(blob)
        detections = net.forward()
        #confidence = detections[0, 0, 0, 2]
        #if confidence >= self.conf_threshold :
        d = 0
        x1 = int(detections[0, 0, d, 3] * frameWidth)
        y1 = int(detections[0, 0, d, 4] * frameHeight)
        x2 = int(detections[0, 0, d, 5] * frameWidth)
        y2 = int(detections[0, 0, d, 6] * frameHeight)
        
        width = x2-x1
        height = y2-y1
        max_size = max(width, height)
        x1, x2 = max(0, (x1+x2)//2 - max_size//2), min(frameWidth, (x1+x2)//2 + max_size//2)
        y1, y2 = max(0, (y1+y2)//2 - max_size//2), min(frameHeight, (y1+y2)//2 + max_size//2)
        if x1<x2 and y1<y2 :
            #X = X[y1:y2, x1:x2, :]
            det = dlib.rectangle(x1, y1, x2, y2)
        else : # face detector fails : use the whole original image
            #print(ID)
            det = dlib.rectangle(0, 0, frameWidth, frameHeight)

        ## Align     
        shape = predictor(X, det)
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)

        M = get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(X, M, (frameWidth, frameHeight), flags=cv2.INTER_CUBIC)
        del M
        cropped = crop_image(rotated, det)
        ## Resize and order
        X = cv2.resize(cropped, (256, 256))
        cv2.imwrite(os.path.join(resultpath, k, img), X)

    
