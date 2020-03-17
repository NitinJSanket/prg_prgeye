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
from Network.HomographyNetICSTNSimpler import  ICSTN
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
import Misc.STNUtils as stn
import Misc.TFUtils as tu
import Misc.warpICSTN as warp
import Misc.warpICSTN2 as warp2
from Misc.DataHandling import *
from Misc.BatchCreationNP import *
from Misc.BatchCreationTF import *


# Don't generate pyc codes
sys.dont_write_bytecode = True         


def SetupAll(ReadPath):
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    DirNames - Full path to all image files without extension
    Train/Val/Test - Idxs of all the images to be used for training/validation (held-out testing in this case)/testing
    Ratios - Ratios is a list of fraction of data used for [Train, Val, Test]
    CheckPointPath - Path to save checkpoints/model
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrain/Val/TestSamples - length(Train/Val/Test)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    Train/Val/TestLabels - Labels corresponding to Train/Val/Test
    """
    # Setup DirNames
    DirNamesPath = ReadPath + os.sep + 'DirNames.txt'
    TestNames = ReadDirNames(DirNamesPath)
    
    # Image Input Shape
    PatchSize = np.array([128, 128, 3])
    ImageSize = np.array([300, 300, 3])
    NumTestSamples = len(TestNames)

    return TestNames, ImageSize, PatchSize, NumTestSamples

def ReadDirNames(DirNamesPath):
    """
    Inputs: 
    Path is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read DirNames file
    DirNames = open(DirNamesPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()

    return DirNames

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # TODO: Make LogDir
    # TODO: Make logging file a parameter
    # TODO: Time to complete print

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/nitin/Datasets/MSCOCO/train2014', help='Base path of images, Default:/home/nitin/Datasets/MSCOCO/train2014')
    Parser.add_argument('--MiniBatchSize', type=int, default=1, help='MiniBatchSize, Default:1')
    Parser.add_argument('--GPUDevice', type=int, default=-1, help='GPUDevice, Default:-1')

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    MiniBatchSize = Args.MiniBatchSize
    GPUDevice = Args.GPUDevice

    tu.SetGPU(GPUDevice)
 
    TestNames, ImageSize, PatchSize, NumTestSamples = SetupAll(BasePath)

    PatchSize = np.array([128, 128, 3])
    count = 0
    IBatch = []
    while(count < MiniBatchSize):
        ImageName = BasePath + os.sep + TestNames[count]
        I = cv2.imread(ImageName)
        if(I is None):
            continue
        count += 1
        I = iu.CenterCrop(I, PatchSize) # np.ones(PatchSize, dtype=np.uint8)*255
        IBatch.append(I)

    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='Input')
    Augmentations =  ['Brightness', 'Contrast', 'Hue', 'Saturation', 'Gamma', 'Gaussian']
    da = iu.DataAugmentationTF(ImgPH, Augmentations)
    DataAug = da.RandPerturbBatch()
    # da = iu.DataAugmentationNP(Augmentations = Augmentations)
    # Timer1 = mu.tic()
    # IPerturbBatch = da.RandPerturbBatch(IBatch)
    # print(mu.toc(Timer1))
    
    # for count in range(MiniBatchSize):
    #     cv2.imshow('I, IPerturb {}/{}'.format(count+1, MiniBatchSize), np.hstack((IBatch[count], IPerturbBatch[count])))
    #     cv2.waitKey(0)
    
    # DataAug = tf.clip_by_value(tf.image.random_brightness(ImgPH, 20), 0.0, 255.0) # tf.image.adjust_brightness(ImgPH, 0.2)
    # DataAug = tf.image.adjust_contrast(ImgPH, contrast_factor=0.6)
    # DataAug = tf.image.adjust_hue(ImgPH, delta=0.4)
    # DataAug = tf.image.adjust_saturation(ImgPH, 5)
    # DataAug = tf.clip_by_value(ImgPH + tf.random.normal(shape = (MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), mean = 0.0, stddev = 20.0), 0.0, 255.0)
    with tf.Session() as sess:
        FeedDict = {ImgPH: IBatch}
        Timer1 = mu.tic()
        IPerturbBatch = sess.run([DataAug], feed_dict=FeedDict)[0]
        print(mu.toc(Timer1))

    IPerturbBatch = IPerturbBatch.astype('uint8')
    for count in range(MiniBatchSize):
        cv2.imshow('I, IPerturb {}/{}'.format(count+1, MiniBatchSize), np.hstack((IBatch[count], IPerturbBatch[count])))
        cv2.waitKey(0)


if __name__ == '__main__':
    main()

