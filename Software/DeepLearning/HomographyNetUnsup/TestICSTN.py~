#!/usr/bin/env python

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)

# TODO:
# Clean print statements
# Global step only loss/epoch on tensorboard
# Print Num parameters in model as a function
# Clean comments
# Check Factor from network list
# ClearLogs command line argument
# Adapt more augmentation from: https://github.com/sthalles/deeplab_v3/blob/master/preprocessing/inception_preprocessing.py
# Tensorboard logging of images

import tensorflow as tf
import cv2
import sys
import os
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.EVHomographyNetUnsup import EVHomographyNetUnsup
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import Misc.PolarTransformer as ptn
import Misc.STNUtils as stn
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

def RandHomographyPerturbation(I, Rho, PatchSize, ImageSize=None, Vis=False):
    """
    Inputs: 
    I is the input image
    Rho is the maximum perturbation in either direction on each corner, i.e., perturbation for each corner lies in [-Rho, Rho]
    Vis when enabled, Visualizes the image and the perturbed image 
    Outputs:
    H is the random homography
    Points are labeled as:
    
    Top Left = p1, Top Right = p2, Bottom Right = p3, Bottom Left = p4 (Clockwise from Top Left)
    Code adapted from: https://github.com/mez/deep_homography_estimation/blob/master/Dataset_Generation_Visualization.ipynb
    """
    '''
    if(ImageSize is None):
        ImageSize = np.shape(I) 
    
    IOrg = I.copy()
    RandX = random.randint(Rho, ImageSize[1]-Rho-PatchSize[1])
    RandY = random.randint(Rho, ImageSize[0]-Rho-PatchSize[0])

    p1 = (RandX, RandY)
    p2 = (RandX, RandY + PatchSize[0])
    p3 = (RandX + PatchSize[1], RandY + PatchSize[0])
    p4 = (RandX + PatchSize[1], RandY)

    AllPts = [p1, p2, p3, p4]

    if(Vis is True):
        IDisp = I.copy()
        cv2.imshow('org', I)
        cv2.waitKey(1)

    if(Vis is True):
        IDisp = I.copy()
        cv2.polylines(IDisp, np.int32([AllPts]), 1, (0,0,0))
        cv2.imshow('a', IDisp)
        cv2.waitKey(1)

    PerturbPts = []
    for point in AllPts:
        PerturbPts.append((point[0] + random.randint(-Rho,Rho), point[1] + random.randint(-Rho,Rho)))

    if(Vis is True):
        PertubImgDisp = I.copy()
        cv2.polylines(PertubImgDisp, np.int32([PerturbPts]), 1, (0,0,0))
        cv2.imshow('b', PertubImgDisp)
        cv2.waitKey(1)
        
    # Obtain Homography between the 2 images
    H = cv2.getPerspectiveTransform(np.float32(AllPts), np.float32(PerturbPts))
    # Get Inverse Homography
    HInv = np.linalg.inv(H)

    WarpedI = cv2.warpPerspective(I, HInv, (ImageSize[1],ImageSize[0]))
    if(Vis is True):
        WarpedImgDisp = WarpedI.copy()
        cv2.imshow('c', WarpedImgDisp)
        cv2.waitKey(1)

    Mask = np.zeros(np.shape(I))
    Mask[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :] = 1
    CroppedI = I[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :]
    CroppedWarpedI = WarpedI[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :]
    
    if(Vis is True):
        CroppedIDisp = np.hstack((CroppedI, CroppedWarpedI))
        print(np.shape(CroppedIDisp))
        cv2.imshow('d', CroppedIDisp)
        cv2.waitKey(0)

    H4Pt = np.subtract(np.array(PerturbPts), np.array(AllPts))
    H4PtCol = np.transpose(np.reshape(H4Pt, (1,np.product(H4Pt.shape))))


    return CroppedI, CroppedWarpedI, H4PtCol, PerturbPts, AllPts, IOrg, WarpedI, Mask
    '''
    if(ImageSize is None):
        ImageSize = np.shape(I) 
    
    RandX = random.randint(Rho, ImageSize[1]-Rho-PatchSize[1])
    RandY = random.randint(Rho, ImageSize[0]-Rho-PatchSize[0])

    p1 = (RandX, RandY)
    p2 = (RandX, RandY + PatchSize[0])
    p3 = (RandX + PatchSize[1], RandY + PatchSize[0])
    p4 = (RandX + PatchSize[1], RandY)

    AllPts = [p1, p2, p3, p4]

    # Rotation Part
    MaxRVal = np.array([2.0, 2.0, 2.0]) # In Degrees, Using 30 fps at 60 deg/s 
    EulAng = 0.05*MaxRVal*([np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5])
    R = Rot.from_euler('zyx', EulAng, degrees=True).as_dcm()
    # R = np.eye(3)

    # Translation Part
    MaxTVal = np.array([[0.08], [0.08], [0.02]]) # In m, Using 30 fps at 2.5 m/s
    T = np.array(2*MaxTVal*([[np.random.rand() - 0.5],[np.random.rand() - 0.5],[np.random.rand() - 0.5]]))
    
    # Normal Part
    N = np.array([[0.0], [0.0], [1.0]]) # Keep this fixed 
    N = np.divide(N, np.linalg.norm(N))

    # Camera Matrix
    K = np.eye(3)
    K[0,0] = 400.0
    K[1,1] = 400.0
    K[0,2] = ImageSize[0]/2
    K[1,2] = ImageSize[1]/2
    
    # Compose Homography
    H = np.add(R, np.matmul(T, N.T))
    H = np.divide(H, H[2,2])
    H = np.matmul(K, np.matmul(H, np.linalg.inv(K)))

    # Get Inverse Homography
    HInv = np.linalg.inv(H)
    
    if(Vis is True):
        IDisp = I.copy()
        cv2.imshow('org', I)
        cv2.waitKey(1)

    if(Vis is True):
        IDisp = I.copy()
        cv2.polylines(IDisp, np.int32([AllPts]), 1, (0,0,0))
        cv2.imshow('a', IDisp)
        cv2.waitKey(1)

    PerturbPts = []
    for point in AllPts:
        PtNow = np.squeeze(np.matmul(H, np.array([[point[0]], [point[1]], [1.0]])))
        PtNow = np.ceil(np.divide(PtNow, PtNow[2]))
        PerturbPts.append((PtNow[0], PtNow[1]))

    if(Vis is True):
        PertubImgDisp = I.copy()
        cv2.polylines(PertubImgDisp, np.int32([PerturbPts]), 1, (0,0,0))
        cv2.imshow('b', PertubImgDisp)
        cv2.waitKey(1)

    WarpedI = cv2.warpPerspective(I, HInv, (ImageSize[1],ImageSize[0]))
    
    if(Vis is True):
        WarpedImgDisp = WarpedI.copy()
        cv2.imshow('c', WarpedImgDisp)
        cv2.waitKey(1)

    Mask = np.zeros(np.shape(I))
    Mask[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :] = 1
    CroppedI = I[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :]
    CroppedWarpedI = WarpedI[RandY:RandY + PatchSize[0], RandX:RandX + PatchSize[1], :]
    

    if(Vis is True):
        CroppedIDisp = np.hstack((CroppedI, CroppedWarpedI))
        print(np.shape(CroppedIDisp))
        cv2.imshow('d', CroppedIDisp)
        cv2.waitKey(0)

    H4Pt = np.subtract(np.array(PerturbPts), np.array(AllPts))
    H4PtCol = np.reshape(H4Pt, (np.product(H4Pt.shape), 1))

    # I is the Original Image I1
    # WarpedI is I2
    # CroppedI is I1 cropped to patch Size
    # CroppedWarpeI is I1 Crop Warped (I2 Crop)
    # AllPts is the patch corners in I1
    # PerturbPts is the patch corners of I2 in I1
    # H4PtCol is the delta movement of corners between AllPts and PerturbPts
    # Mask is the active region of I1Patch in I1
    return CroppedI, CroppedWarpedI, H4PtCol, PerturbPts, AllPts, I, WarpedI, Mask
    

def main():
    BasePath = '/home/nitin/Datasets/MSCOCO/train2014/COCO_train2014_000000484344.jpg'
    I = cv2.imread(BasePath)
    # cv2.circle(I, (np.shape(I)[0]/2, np.shape(I)[1]/2), 1, (0,0,255), -1)
    # Refer to: https://github.com/opencv/opencv/blob/master/samples/python/logpolar.py
    # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
    # P = cv2.logPolar(I, (np.shape(I)[0]/2, np.shape(I)[1]/2), 120, cv2.WARP_FILL_OUTLIERS)
    
    # cv2.imshow('IOrg', I)
    # cv2.imshow('a', P)
    # cv2.waitKey(0)
    
    # I = np.hsplit(I, 2)
    # I = I[0]
    
    PatchSize = [128, 128, 3]
    Rho = 20
    MiniBatchSize = 1
    I, IPerturb, H4Pt, PerturbPts, AllPts, IOrg, WarpedI, Mask = RandHomographyPerturbation(I, Rho, PatchSize, Vis=False)
    P1 = cv2.logPolar(I, (np.shape(I)[0]/2, np.shape(I)[1]/2), 20, cv2.WARP_FILL_OUTLIERS)
    P2 = cv2.logPolar(IPerturb, (np.shape(IPerturb)[0]/2, np.shape(IPerturb)[1]/2), 20, cv2.WARP_FILL_OUTLIERS)

    cv2.imshow('I1', I)
    cv2.imshow('I2', IPerturb)
    cv2.imshow('P1', P1)
    cv2.imshow('P2', P2)
    cv2.waitKey(0)
       
    
    '''
    cv2.circle(I,(64, 64), 1, (0,0,255), -1)
    
    AllPts = np.expand_dims(AllPts, axis=0)
    PerturbPts = np.expand_dims(PerturbPts, axis=0)
    IOrg = np.expand_dims(IOrg, axis=0)
    I = np.expand_dims(I, axis=0)
    Mask = np.expand_dims(Mask, axis=0)
    AllPtsTensor = tf.convert_to_tensor(np.float32(AllPts), dtype='float')
    PerturbPtsTensor = tf.convert_to_tensor(np.float32(PerturbPts), dtype='float')
    H4PtTensor = tf.convert_to_tensor(np.float32(H4Pt), dtype='float')
    ITensor = tf.convert_to_tensor(np.float32(I), dtype='float')
    MaskTensor = tf.convert_to_tensor(np.float32(Mask), dtype='float')
    theta = tf.convert_to_tensor(np.float32([[0], [0]]), dtype='float')
    out_size = [MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]]

    # BM = tf.boolean_mask(ITensor, MaskTensor)
    # BM = tf.reshape(BM, (MiniBatchSize, 128, 128, 3))
    Polar = ptn.polar_transformer(ITensor, theta, out_size, name='polar_transformer', log=True, radius_factor=0.5)
    
    # MiniBatchSize = 1
    # HMat = stn.solve_DLT(MiniBatchSize, AllPtsTensor, H4PtTensor)
    # Tensorflow's direct without scaling as in Unsupervised Deep Homography Paper
    # warped_image = tf.contrib.image.transform(ITensor, HMat, interpolation='BILINEAR', output_shape=None, name=None)
    # WarpMask = tf.contrib.image.transform(MaskTensor, HMat, interpolation='BILINEAR', output_shape=None, name=None)
    # Tensorflow's direct with scaling as in Unsupervised Deep Homography Paper
    # WarpedIPred = stn.transform(out_size, HMat, MiniBatchSize, ITensor)
    # WarpMask = stn.transform(out_size, HMat, MiniBatchSize, MaskTensor)
    # Try Polar Transformer Network
    with tf.Session() as sess:
        # A = WarpedIPred.eval()
        # H = HMat.eval()
        # Z = WarpMask.eval()
        # print(H)
        # print(np.shape(A))
        # A = np.squeeze(Polar.eval())
        A = np.squeeze(Polar.eval())
        # cv2.namedWindow('a',  cv2.WINDOW_NORMAL)
        # cv2.imshow('aaaaa', np.hstack((np.squeeze(A),  WarpedI, IOrg[0], np.squeeze(Z))))
        # plt.imshow(np.hstack((np.squeeze(A), np.squeeze(IOrg))))
        # plt.show()
        #  = np.zeros(np.shape(np.squeeze(IOrg)))
        # B = cv2.normalize(A,B,0,255,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.imshow('a', np.hstack((np.uint8(A), np.squeeze(IOrg))))
        cv2.imshow('IOrg', I[0])
        cv2.imshow('a', np.uint8(A))
        print(np.shape(A))
        print(np.amax(A))
        print(np.amin(A))
        cv2.waitKey(0)



    # Try Sparse Image Warp
    # warped_image, flow_field = tf.contrib.image.sparse_image_warp(image=ITensor, source_control_point_locations=AllPtsTensor,\
    #     dest_control_point_locations=PerturbPtsTensor, \
    #    interpolation_order=1, regularization_weight=0.0,  num_boundary_points=0,  name='sparse_image_warp')
    # with tf.Session() as sess:
    #    A = warped_image.eval()
    # cv2.imshow('a', warped_image)
    # cv2.waitKey(0)
    '''

if __name__ == '__main__':
    main()
 
