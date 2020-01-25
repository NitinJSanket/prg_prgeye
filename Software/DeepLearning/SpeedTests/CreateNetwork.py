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
from Network.EVHomographyNetUnsupSmall import EVHomographyNetUnsupSmall
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
import Misc.STNUtils as stn
import Misc.TFUtils as tu
import Misc.MiscUtils as mu

# Don't generate pyc codes
sys.dont_write_bytecode = True
    
def GenerateModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, MiniBatchSize):    
    # Predict output with forward pass
    if(NetworkType == 'Small'):
        prHVal = EVHomographyNetUnsupSmall(ImgPH, ImageSize, MiniBatchSize)
    elif(NetworkType == 'Large'):
        prHVal = EVHomographyNetUnsup(ImgPH, ImageSize, MiniBatchSize)
            
    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:       
        sess.run(tf.global_variables_initializer())
        print('New model initialized....')

        # Print Number of parameters in the network    
        tu.FindNumParams(1)
        
        # Save model every epoch
        SaveName = CheckPointPath + os.sep + ModelPrefix +'model.ckpt'
        Saver.save(sess, save_path=SaveName)
        print(SaveName + ' Model Saved...')

    # Reset graph after using: https://github.com/tensorflow/tensorflow/issues/19731
    tf.reset_default_graph()

def SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, MiniBatchSize, TestMultipleMiniBatch, NumTest, MiniBatchSizeIncrement, MiniBatchMin, Append=False):
    if(TestMultipleMiniBatch):
        StartIdx = MiniBatchMin - 1
    else:
        StartIdx =  MiniBatchSize - 1 # Run 1 iteration only

    # Open file for logging
    if(not Append):
        LogFile = open(CheckPointPath + os.sep + ModelPrefix + 'Log.txt', 'w')
    else:
        LogFile = open(CheckPointPath + os.sep + ModelPrefix + 'Log.txt', 'a')

    for count2 in range(StartIdx, MiniBatchSize, MiniBatchSizeIncrement):
        MiniBatchSizeNow = count2 + 1
        print('Testing MiniBatchSize {} ....'.format(MiniBatchSizeNow))
        ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSizeNow, ImageSize[0], ImageSize[1], 2*ImageSize[2]), name='Input')
            
        # Predict output with forward pass
        if(NetworkType == 'Small'):
            prHVal = EVHomographyNetUnsupSmall(ImgPH, ImageSize, MiniBatchSize)
        elif(NetworkType == 'Large'):
            prHVal = EVHomographyNetUnsup(ImgPH, ImageSize, MiniBatchSize)

        # Setup Saver
        Saver = tf.train.Saver()

        with tf.Session() as sess:       
            sess.run(tf.global_variables_initializer())
            print('Model initialized....')

            # Write this the first time only
            if((count2 == StartIdx) and (not Append)):
                # Print Number of parameters in the network    
                NumParams = tu.FindNumParams(1)
                # Print Number of FLOPs in the network
                NumFLOPs = tu.FindNumFlops(sess, 1)

                # Write values to file
                LogFile.write('NumParams {}, NumFLOPs {}\n'.format(NumParams, NumFLOPs))
                LogFile.write('MiniBatchSize, Avg. time (s), Eff. Frame Rate (fps) \n')
            
            # Load model
            LoadName = CheckPointPath + os.sep + ModelPrefix +'model.ckpt'
            Saver.restore(sess, LoadName)
            # Extract only numbers from the name
            print('Loaded weights from ' + LoadName + '....')

            # Run once to setup graph
            RandImg = np.random.rand(ImageSize[0], ImageSize[1], 2*ImageSize[2])
            RandImg = RandImg[np.newaxis, :, :, :]

            RandImgBatch = np.tile(RandImg, (MiniBatchSizeNow,1,1,1))
            for count in range(NumTest+1):
                if(count == 1):
                    # Start timer after first iteration
                    Timer1 = mu.tic()
                FeedDict = {ImgPH: RandImgBatch}
                _ = sess.run([prHVal], FeedDict)

        Time = mu.toc(Timer1)/NumTest
        FPS = 1/Time*MiniBatchSizeNow
        LogFile.write('{}, {}, {}\n'.format(MiniBatchSizeNow, Time, FPS))
        print('Testing MiniBatchSize {} Complete....'.format(MiniBatchSizeNow))
        # Reset graph after using: https://github.com/tensorflow/tensorflow/issues/19731
        tf.reset_default_graph()

    LogFile.close()

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
    Parser.add_argument('--CheckPointPath', default='/home/nitin/PRGEye/CheckPoints/SpeedTests', help='Path to save Model, Default:/home/nitin/PRGEye/CheckPoints/SpeedTests')
    Parser.add_argument('--ModelPrefix', default='', help='Prefix for model name, Default: ''')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--NetworkType', default='Large', help='Choice of Network type, choose from Small, Large, Default:Large')
    Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--TestMultipleMiniBatch', type=int, default=0, help='Used for Testing only, if active MiniBatchSize represents max value')
    Parser.add_argument('--NumTest', type=int, default=100, help='Number of times code will run to take avg. time')
    Parser.add_argument('--Mode', default='Train', help='Mode: Train or Test or TrainTest')
    Parser.add_argument('--MiniBatchMin', type=int, default=1, help='Used for Testing only, is active when TestMultipleMiniBatch is used and MiniBatchMin represents min value.')
    Parser.add_argument('--MiniBatchSizeIncrement', type=int, default=1, help='Used for Testing only, is active when TestMultipleMiniBatch is used and MiniBatchSize represents max value. The code is tested for these increment steps. ')
    Parser.add_argument('--ForceBatchSize1', type=int, default=0, help='Used for Testing only, is active when TestMultipleMiniBatch is used and MiniBatchSize represents max value. Force runs for BatchSize of 1.')
    

    Args = Parser.parse_args()
    CheckPointPath = Args.CheckPointPath
    ModelPrefix = Args.ModelPrefix
    GPUDevice = Args.GPUDevice
    NetworkType = Args.NetworkType
    MiniBatchSize = Args.MiniBatchSize
    Mode = Args.Mode
    TestMultipleMiniBatch = bool(Args.TestMultipleMiniBatch)
    NumTest = Args.NumTest
    MiniBatchSizeIncrement = Args.MiniBatchSizeIncrement
    MiniBatchMin = Args.MiniBatchMin
    ForceBatchSize1 = bool(Args.ForceBatchSize1)

    # Parameters
    ImageSize = np.array([128, 128, 3])
    
    # Set GPUDevice
    tu.SetGPU(GPUDevice)

    # Placeholders
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 2*ImageSize[2]), name='Input')
    
    # Generate the model and save        
    if(Mode == 'Train'):
        GenerateModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, MiniBatchSize)
    elif(Mode == 'Test'):
        if(ForceBatchSize1 and TestMultipleMiniBatch):
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, 1, False, NumTest, MiniBatchSizeIncrement, 1)
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, MiniBatchSize, TestMultipleMiniBatch, NumTest, MiniBatchSizeIncrement, MiniBatchMin, Append=True)
        else:
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, MiniBatchSize, TestMultipleMiniBatch, NumTest, MiniBatchSizeIncrement, MiniBatchMin)
    elif(Mode=='TrainTest'):
        GenerateModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, MiniBatchSize)
        if(ForceBatchSize1 and TestMultipleMiniBatch):
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, 1, False, NumTest, MiniBatchSizeIncrement, 1)
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, MiniBatchSize, TestMultipleMiniBatch, NumTest, MiniBatchSizeIncrement, MiniBatchMin, Append=True)
        else:
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, NetworkType, MiniBatchSize, TestMultipleMiniBatch, NumTest, MiniBatchSizeIncrement, MiniBatchMin)
    else:
        print('ERROR: Invalid Mode!')
        sys.exit(0)
        
    
if __name__ == '__main__':
    main()
