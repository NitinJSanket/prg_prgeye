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

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(ReadPath, warpType):
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

    # Similarity Perturbation Parameters
    MaxParams = np.array([0.5, 0.4, 0.4]) # np.array([0.25, 0.2, 0.2]) # np.array([0.5, 0.4, 0.4])
    HObj = iu.HomographyICTSN(TransformType = 'pseudosimilarity', MaxParams = MaxParams)
    
    return TestNames, ImageSize, PatchSize, NumTestSamples, MaxParams, HObj

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


class BatchGeneration():
    def __init__(self, sess, WarpI1PatchIdealGen, IOrgPH, HPH):
        self.sess = sess
        self.WarpI1PatchIdealGen = WarpI1PatchIdealGen
        self.IOrgPH = IOrgPH
        self.HPH = HPH
        
    def RandSimilarityPerturbationTF(self, I1, HObj, PatchSize, MiniBatchSize, ImageSize=None, Vis = False):
        if(ImageSize is None):
            ImageSize = np.array(np.shape(I1))[1:]
            # TODO: Extract MiniBatchSize here

        H, Params = HObj.GetRandReducedHICSTN()

        # Maybe there is a better way? https://dominikschmidt.xyz/tensorflow-data-pipeline/
        
        FeedDict = {self.IOrgPH: I1, self.HPH: H}
        I2 = np.uint8(self.sess.run([self.WarpI1PatchIdealGen], feed_dict=FeedDict)[0]) # self.WarpI1PatchIdealGen.eval(feed_dict=FeedDict)

        # Crop in center for PatchSize
        P1 = iu.CenterCrop(I1, PatchSize)
        P2 = iu.CenterCrop(I2, PatchSize)

        if(Vis is True):
            for count in range(MiniBatchSize):
                A = I1[count]
                B = I2[count]
                AP = P1[count]
                BP = P2[count]
                cv2.imshow('I1, I2', np.hstack((A, B)))
                cv2.imshow('P1, P2', np.hstack((AP, BP)))
                cv2.waitKey(0)

        # P1 is I1 cropped to patch Size
        # P2 is I1 Crop Warped (I2 Crop)
        # H is Homography
        # Params is the stuff H is made from 
        return I1, I2, P1, P2, H, Params


    def GenerateBatchTF(self, I, PatchSize, MiniBatchSize, HObj, BasePath, ImageSize, Vis = False):
        """
        Inputs: 
        DirNames - Full path to all image files without extension
        NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
        TrainLabels - Labels corresponding to Train
        NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
        ImageSize - Size of the Image
        MiniBatchSize is the size of the MiniBatch
        Outputs:
        I1Batch - Batch of I1 images after standardization and cropping/resizing to ImageSize
        HomeVecBatch - Batch of Homing Vector labels
        """

        # Similarity and Patch generation 
        I1Batch, I2Batch, P1Batch, P2Batch, HBatch, ParamsBatch = self.RandSimilarityPerturbationTF(I, HObj, PatchSize, MiniBatchSize, ImageSize = None, Vis = Vis)
            
        ICombined = np.concatenate((P1Batch[:,:,:,0:3], P2Batch[:,:,:,0:3]), axis=3)
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IBatch = iu.StandardizeInputs(np.float32(ICombined))

        return IBatch, I1Batch, I2Batch, P1Batch, P2Batch, HBatch, ParamsBatch



def TestOperation(PatchPH, I1PH, I2PH, PerturbParamsPH, PerturbHPH, ImageSize, PatchSize, ModelPath, ReadPath,\
                  WritePath, TestNames, NumTestSamples, CropType, MiniBatchSize, MaxParams, warpType, HObj, opt, optdg, Net, InitNeurons):
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
    # Data Generation
    I2Gen = warp2.transformImage(optdg, I1PH, PerturbHPH)
    # Predict output with forward pass
    # Create Network Object with required parameters
    VN = Net.ResNet(InputPH = PatchPH, Training = True, Opt = opt, InitNeurons = InitNeurons)
    # Predict output with forward pass
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
        PredOuts = open(WritePath + os.sep + 'PredOutsLargeDeviation.txt', 'w') # LargeDeviation
        PredOuts.write('Model Used: {}\n'.format(ModelPath))
        PredOuts.write('Model Statistics: \n')
        PredOuts.write('Number of Parameters: {}\n'.format(NumParams))
        PredOuts.write('Number of Flops: {}\n'.format(NumFlops))
        PredOuts.write('Expected Model Size: {} MB\n'.format(ModelSize))
        PredOuts.write('Dataset Used to test: {}\n'.format(ReadPath))
        PredOuts.write('Max Params: {} using {} warping \n'.format(MaxParams, warpType[-1]))
        PredOuts.write('ImageName' + '\t' + 'ParamsBatch' + '\t' + 'prParamsVal' + '\t' +\
                           'ErrorScalePxPred' + '\t' + 'ErrorScalePxIdentity' + '\t' +\
                           'ErrorTransPxPred' + '\t' + 'ErrorTransPxIdentity' + '\n')

        # Create Write Folder if doesn't exist
        if(not os.path.exists(WritePath)):
            os.makedirs(WritePath)

        # Create Batch Generator Object
        bg = BatchGeneration(sess, I2Gen, I1PH, PerturbHPH)

        for TestName in tqdm(TestNames):
            # Generate batch of I1 original images
            ImageName = ReadPath + os.sep + TestName
            I = cv2.imread(ImageName)
            I = iu.CenterCrop(I, ImageSize)
            if (I is None):
                continue
            I = np.array(I[np.newaxis, :, :, :])
            IBatch, I1Batch, I2Batch, P1Batch, P2Batch, HBatch, ParamsBatch = bg.GenerateBatchTF(I, PatchSize, MiniBatchSize, HObj, ReadPath, ImageSize, Vis = False)

            # TODO: Better way is to feed data into a MiniBatch and Extract it again
            FeedDict = {VN.InputPH: IBatch}
            prHVal, prParamsVal = sess.run([prH, prParams], feed_dict=FeedDict)

            # Extract Values
            prHVal = prHVal[0]
            prParamsVal = prParamsVal[0]
            # WarpI1PatchVal = WarpI1PatchVal[0]
            ParamsBatch = ParamsBatch[0]

            def ComputeScaleError(ImageSize, GT, Pred = None):
                if(Pred is None):
                    Pred = np.array(0.)
                ErrorScale = np.abs(GT - Pred)
                ErrorScalePx = ErrorScale*np.sqrt((ImageSize[0]/2)**2 + (ImageSize[1]/2)**2)
                return ErrorScale, ErrorScalePx

            def ComputeTransError(ImageSize, GT, Pred = None):
                if(Pred is None):
                    Pred = np.array(np.zeros((2,1)))
                ErrorTransX = GT[0] - Pred[0]
                ErrorTransY = GT[1] - Pred[1]
                ErrorTransXPx = ErrorTransX*(ImageSize[0]/2)
                ErrorTransYPx = ErrorTransY*(ImageSize[1]/2)
                ErrorTrans = np.sqrt(ErrorTransX**2 + ErrorTransY**2)
                ErrorTransPx = np.sqrt(ErrorTransXPx**2 + ErrorTransYPx**2)
                return ErrorTrans, ErrorTransPx

            # Compute Error between GT and Pred
            ErrorScalePred, ErrorScalePxPred = ComputeScaleError(PatchSize, ParamsBatch[0], prParamsVal[0])
            ErrorTransPred, ErrorTransPxPred = ComputeTransError(PatchSize, ParamsBatch[1:], prParamsVal[1:])

            # Compute Identity Error
            ErrorScaleIdentity, ErrorScalePxIdentity = ComputeScaleError(PatchSize, ParamsBatch[0])
            ErrorTransIdentity, ErrorTransPxIdentity = ComputeTransError(PatchSize, ParamsBatch[1:])

            PredOuts.write(ImageName + '\t' +  str(ParamsBatch) + '\t' +  str(prParamsVal) + '\t' + \
                           str(ErrorScalePxPred) +  '\t' + str(ErrorScalePxIdentity) + '\t' + \
                           str(ErrorTransPxPred) +  '\t' + str(ErrorTransPxIdentity) + '\n')
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

    # Setup all needed parameters including file reading
    InitNeurons = 13
    warpType = ['translation', 'translation', 'scale', 'scale'] # ['translation', 'translation', 'scale', 'scale'] # ['scale', 'scale', 'translation', 'translation'] # ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity'] # ['scale', 'scale', 'translation', 'translation'] # ['pseudosimilarity', 'pseudosimilarity']
    # Homography Perturbation Parameters
    TestNames, ImageSize, PatchSize, NumTestSamples, MaxParams, HObj = SetupAll(ReadPath, warpType)

    opt = warp2.Options(PatchSize=PatchSize, MiniBatchSize=MiniBatchSize, warpType = warpType) # ICSTN Options
    optdg = warp2.Options(PatchSize=ImageSize, MiniBatchSize=MiniBatchSize, warpType = [warpType[-1]]) # Data Generation Options, warpType should the same the last one in the previous command
     
    # Define PlaceHolder variables for Input and Predicted output
    PatchPH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], PatchSize[2]*2), name='Input')
    I1PH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]), name='I1')
    I2PH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]), name='I2')
    PerturbParamsPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 3), name='PerturbParams')
    PerturbHPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 3, 3), name='PerturbH')

    if(not os.path.exists(WritePath)):
        cprint("WARNING: %s doesnt exist, Creating it."%WritePath, 'yellow')
        os.mkdir(WritePath)

    TestOperation(PatchPH, I1PH, I2PH, PerturbParamsPH, PerturbHPH, ImageSize, PatchSize, ModelPath,\
                  ReadPath, WritePath, TestNames, NumTestSamples, CropType, MiniBatchSize, MaxParams, warpType, HObj, opt, optdg, Net, InitNeurons)

     
if __name__ == '__main__':
    main()
 
