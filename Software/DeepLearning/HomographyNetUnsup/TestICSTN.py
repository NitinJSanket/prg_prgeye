
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
import Misc.warpICSTN as warp
import Misc.MiscUtils as mu

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

    if(ImageSize is None):
        ImageSize = np.shape(I) 
    
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

    # Multiply by M and Minv
    M = np.eye(3)
    M[0,0] = PatchSize[0]/2
    M[0,2] = PatchSize[0]/2
    M[1,1] = PatchSize[1]/2
    M[1,2] = PatchSize[1]/2
    H = np.matmul(np.matmul(M, H), np.linalg.inv(M))
    
    # Normalize by H(2,2)
    H = np.divide(H, H[2,2])

    # Extract first 8 elements
    H8El = np.ndarray.flatten(H)
    H8El = H8El[0:8]

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

    # H4Pt = np.subtract(np.array(PerturbPts), np.array(AllPts))
    # H4PtCol = np.reshape(H4Pt, (np.product(H4Pt.shape), 1))

    # I is the Original Image I1
    # WarpedI is I2
    # CroppedI is I1 cropped to patch Size
    # CroppedWarpeI is I1 Crop Warped (I2 Crop)
    # AllPts is the patch corners in I1
    # PerturbPts is the patch corners of I2 in I1
    # H8El is the first 8 elements of the Homography matrix from AllPts to PerturbPts
    # Mask is the active region of I1Patch in I1
    return I, WarpedI, CroppedI, CroppedWarpedI, AllPts, PerturbPts, H8El, Mask  

def main():
    BasePath = '/home/nitin/Datasets/MSCOCO/train2014/COCO_train2014_000000484344.jpg'
    I = cv2.imread(BasePath)
    
    PatchSize = [128, 128, 3]
    Rho = 20
    MiniBatchSize = 16
    
    class Options:
        def __init__(self, PatchSize=[128,128,3], MiniBatchSize=16, warpType='homography', pertScale=0.25, transScale=0.25):
            self.W = PatchSize[0] # PatchSize is Width, Height, NumChannels
            self.H = PatchSize[1] 
            self.batchSize = MiniBatchSize
            self.warpType = 'homography'
            if self.warpType == 'translation':
                self.warpDim = 2
            elif self.warpType == 'similarity':
                self.warpDim = 4
            elif self.warpType == 'affine':
                self.warpDim = 6
            elif self.warpType == 'homography':
                self.warpDim = 8
            self.canon4pts = np.array([[-1,-1],[-1,1],[1,1],[1,-1]],dtype=np.float32)
	    self.image4pts = np.array([[0,0],[0,PatchSize[1]-1],[PatchSize[0]-1,PatchSize[1]-1],[PatchSize[0]-1,0]],dtype=np.float32)
	    self.refMtrx = warp.fit(Xsrc=self.canon4pts, Xdst=self.image4pts)
            self.pertScale = pertScale
            self.transScale = transScale

    opt = Options()

    pRand = warp.genPerturbationsNP(opt)
    pMtrx, transMtrx = warp.vec2mtrxNP(opt, pRand)

    I, IPerturb, H4Pt, PerturbPts, AllPts, IOrg, WarpedI, Mask = RandHomographyPerturbation(I, Rho, PatchSize, pMtrx, Vis=True)
    input('q')

    ITensor = tf.convert_to_tensor(np.float32(I), dtype='float')
    p = tf.to_float([[0,0,0.1,0,0,0.1,0,0]]) 
    pMtrx = warp.vec2mtrx(opt,p)
    imageWarp = warp.transformImage(opt, ITensor, pMtrx)

    with tf.Session() as sess:
        A = imageWarp.eval()
    cv2.imshow('IOrg', I)
    cv2.imshow('a', np.uint8(mu.remap(A[0], 0.0, 255.0, np.amin(A[0]), np.amax(A[0]))))
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main()
 
