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
import Misc.MiscUtils as mu
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
import Misc.warpICSTN2 as warp2
from Misc.DataHandling import *
from Misc.BatchCreationNP import *
from Misc.BatchCreationTF import *
from Misc.Decorators import *
# Import of network is done in main code
import importlib

# TODO: Add warning for overwriting

# Don't generate pyc codes
sys.dont_write_bytecode = True


def TestOperation(PatchPH, PatchSize, ModelPath,\
                  ReadPath, WritePath, CropType, MiniBatchSize, warpType, opt, Net, Args):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    HomingVecPH is the ground truth  homing vector placeholder
    NumTrainSamples - length(Train)
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    # Test Set Params
    NumImgs = 1000
    StartNum = 0
    SkipNum = Args.SkipNum
    ImgFormat = '.png'
    # Predict output with forward pass
    # Create Network Object with required parameters
    ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
    Network = getattr(Net, ClassName)
    VN = Network(InputPH = PatchPH, Training = True, Opt = opt, InitNeurons = Args.InitNeurons, Suffix = '')
    # Predict output with forward pass
    # WarpI1Patch contains warp of both I1 and I2, extract first three channels for useful data
    prH, prParams, _ = VN.Network()

    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # Restore Model
        Saver.restore(sess, ModelPath)
        # Print out Number of parameters
        NumParams = tu.FindNumParams(1)
        # Print out Number of Flops
        NumFlops = tu.FindNumFlops(sess, 1)
        # Print out Expected Model Size
        ModelSize = tu.CalculateModelSize()*3 # For some wierd reason result has to be multiplied by 3
        print('Expected Model Size is %f' % ModelSize)

        # Create PredOuts File
        Name = 'NetworkConfig.txt'
        NetworkConfig = open(WritePath + os.sep + Name, 'w') # LargeDeviation
        NetworkConfig.write('Model Used: {}\n'.format(ModelPath))
        NetworkConfig.write('Model Statistics: \n')
        NetworkConfig.write('Number of Parameters: {}\n'.format(NumParams))
        NetworkConfig.write('Number of Flops: {}\n'.format(NumFlops))
        NetworkConfig.write('Expected Model Size: {} MB\n'.format(ModelSize))
        NetworkConfig.write('Dataset Used to test: {}\n'.format(ReadPath))
        NetworkConfig.write('ImageName' + '\t' + 'prParamsVal' +'\n')

        Name = 'PredOuts.txt'
        PredOuts = open(WritePath + os.sep + Name, 'w') # LargeDeviation


        # Create Write Folder if doesn't exist
        if(not os.path.exists(WritePath)):
            os.makedirs(WritePath)

        for TestNum in tqdm(range(StartNum, NumImgs-SkipNum, SkipNum)):
            # Generate batch of I1 original images
            ImageName = ReadPath + os.sep +  'frame' + '%06d'%TestNum + ImgFormat
            ImagePairName = ReadPath + os.sep +  'frame' + '%06d'%(TestNum + SkipNum) + ImgFormat
            I1 = cv2.imread(ImageName)
            I2 = cv2.imread(ImagePairName)
            P1 = iu.CenterCrop(I1, PatchSize)
            P2 = iu.CenterCrop(I2, PatchSize)
            if (P1 is None or P2 is None):
                continue
            P1 = np.array(P1[np.newaxis, :, :, :])
            P2 = np.array(P2[np.newaxis, :, :, :])

            ICombined = np.concatenate((P1[:,:,:,0:3], P2[:,:,:,0:3]), axis=3)
            # Normalize Dataset
            # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
            IBatch = iu.StandardizeInputs(np.float32(ICombined))
            
            # TODO: Better way is to feed data into a MiniBatch and Extract it again
            FeedDict = {VN.InputPH: IBatch}
            prHVal, prParamsVal = sess.run([prH, prParams], feed_dict=FeedDict)

            # Extract Values
            prHVal = prHVal[0]
	    prParamsVal = prParamsVal[0]
	
            # PredOuts.write(ImageName + '\t' +  str(prParamsVal) + '\n')
            PredOuts.write(str(prParamsVal[0]) + ',' + str(prParamsVal[1]) + ',' + str(prParamsVal[2]) + '\n')
        PredOuts.close()
                    
                    
def main():
    """
    Inputs: 
    None
    Outputs:
    Runs Testing code
    """
    # TODO: Make LogDir
    # TODO: Display time to end and cleanup other print statements with color
    # TODO: Make logging file a parameter

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/media/nitin/Research/EVDodge/CheckpointsDeblurHomographyLR1e-4Epochs400/399model.ckpt',\
                                                         help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--ReadPath', dest='ReadPath', default='/media/nitin/Research/EVDodge/DatasetChethanEvents/Deblurred',\
                                                                             help='Path to load images from, Default:ReadPath')
    Parser.add_argument('--WritePath', dest='WritePath', default='/media/nitin/Research/EVDodge/DatasetChethanEvents/DeblurredHomography',\
                                                                             help='Path to load images from, Default:WritePath')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--CropType', dest='CropType', default='C', help='What kind of crop do you want to perform? R: Random, C: Center, Default: C')
    Parser.add_argument('--NetworkName', default='Network.ResNet3', help='Name of network file, Default: Network.VanillaNet')
    Parser.add_argument('--InitNeurons', type=int, default=13, help='Number of Init Neurons, Default: 13')
    Parser.add_argument('--SkipNum', type=int, default=2, help='Number of Frames to Skip, Default: 2')

    # Parser.add_argument('--ImageFormat', default='.jpg', help='Image format, default: .jpg')
    # Parser.add_argument('--Prefix', default='COCO_test2014_%012d', help='Image name prefix, default: COCO_test2014_%012d')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    ReadPath = Args.ReadPath
    WritePath = Args.WritePath
    GPUDevice = Args.GPUDevice
    CropType = Args.CropType
    MiniBatchSize = 1
    NetworkName = Args.NetworkName

    # ImageFormat = Args.ImageFormat
    # Prefix = Args.Prefix

    # Import Network Module
    Net = importlib.import_module(NetworkName)
    
    # Set GPUNum
    tu.SetGPU(GPUDevice)

    # Image Input Shape
    PatchSize = np.array([128, 128, 3])


    # Setup all needed parameters including file reading
    warpType = ['translation', 'translation', 'scale', 'scale'] # ['pseudosimilarity'] # ['translation', 'translation', 'scale', 'scale'] # ['scale', 'scale', 'translation', 'translation'] # ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity'] # ['scale', 'scale', 'translation', 'translation'] # ['pseudosimilarity', 'pseudosimilarity']
    # Homography Perturbation Parameters

    opt = warp2.Options(PatchSize=PatchSize, MiniBatchSize=MiniBatchSize, warpType = warpType) # ICSTN Options
     
    # Define PlaceHolder variables for Input and Predicted output
    PatchPH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], PatchSize[2]*2), name='Input')
    # I1PH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]), name='I1')
    # I2PH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]), name='I2')

    if(not os.path.exists(WritePath)):
        cprint("WARNING: %s doesnt exist, Creating it."%WritePath, 'yellow')
        os.mkdir(WritePath)

    TestOperation(PatchPH, PatchSize, ModelPath,\
                  ReadPath, WritePath, CropType, MiniBatchSize, warpType, opt, Net, Args)

     
if __name__ == '__main__':
    main()
 
