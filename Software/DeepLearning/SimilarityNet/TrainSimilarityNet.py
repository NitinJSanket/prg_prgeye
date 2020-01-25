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


# Don't generate pyc codes
sys.dont_write_bytecode = True

def RandSimilarityPerturbation(I1, HObj, PatchSize, ImageSize=None, Vis=False):
    if(ImageSize is None):
        ImageSize = np.shape(I1)

    H, Compositions = HObj.GetRandReducedH(TransformType = ['Yaw', 'Scale', 'T2D'], ScaleToPx = False)
    # ScaleMtrx = np.eye(3) # Scales from [-1, 1] ImageCoordinates to Actual Image Coordinates
    # ScaleMtrx[0,0] = PatchSize[1]/2
    # ScaleMtrx[0,2] = PatchSize[1]/2
    # ScaleMtrx[1,1] = PatchSize[0]/2
    # ScaleMtrx[1,2] = PatchSize[0]/2
    # H = np.matmul(H, np.linalg.inv(ScaleMtrx))

    # I2 = cv2.warpPerspective(I1, np.matmul(ScaleMtrx, H), (ImageSize[1], ImageSize[0])) 
    I2 = cv2.warpPerspective(I1, H, (ImageSize[1], ImageSize[0])) 
    
    # Crop in center for PatchSize
    P1 = iu.CenterCrop(I1, PatchSize)
    P2 = iu.CenterCrop(I2, PatchSize)
    
    if(Vis is True):
        cv2.imshow('I1, I2', np.hstack((I1, I2)))
        cv2.imshow('I1, I2', np.hstack((P1, P2)))
        cv2.waitKey(0)
    
 
    # P1 is I1 cropped to patch Size
    # P2 is I1 Crop Warped (I2 Crop)
    # H is Homography
    # Compositions is the stuff H is made from 
    return I1, I2, P1, P2, H, Compositions

    
def GenerateBatch(TrainNames, PatchSize, MiniBatchSize, HObj, BasePath, OriginalImageSize):
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
    IBatch = [] # P1, P2
    I1Batch = []
    I2Batch = []
    HBatch = []
    CompositionsBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainNames)-1)
        RandImageName = BasePath + os.sep + TrainNames[RandIdx] 
        I = cv2.imread(RandImageName)
        I = iu.RandomCrop(I, OriginalImageSize)
        if (I is None):
            continue
        ImageNum += 1

        # Similarity and Patch generation 
        I1, I2, P1, P2, H, Compositions = RandSimilarityPerturbation(I, HObj, PatchSize, ImageSize=None, Vis=False)
        # [Scale, Yaw, Shear, T2D]
        CompositionsExtracted = np.array([Compositions[0][0]-1, Compositions[3][0], Compositions[3][0]]) # Neglect the stuff you don't need
        
        ICombined = np.dstack((P1[:,:,0:3], P2[:,:,0:3]))
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IS = iu.StandardizeInputs(np.float32(ICombined))

        # Append All Images and Mask
        IBatch.append(IS)
        I1Batch.append(P1)
        I2Batch.append(P2)
        HBatch.append(H)
        CompositionsBatch.append(CompositionsExtracted)

    CompositionsBatch = np.squeeze(CompositionsBatch)
    return IBatch, I1Batch, I2Batch, HBatch, CompositionsBatch              


def LossFunc(I1PH, I2PH, LabelPH, prHVal, prVal, MiniBatchSize, PatchSize, opt):
    # TODO: Warp using A H Ainv
    WarpI1Patch = warp2.transformImage(opt, I1PH, prHVal)
    # L2 loss between predicted and ground truth parameters
    # DiffImg = WarpI1Patch - I2PH
    # Label = warp2.mtrx2vec(opt, LabelPH)
    lossPhoto = tf.reduce_mean(tf.square(prVal - LabelPH))

    # TODO: Use stop gradient to calculate loss of individual components
    # TODO: Try using Normal STN

    # Unsupervised L1 Photometric Loss
    # lossPhoto = tf.reduce_mean(tf.abs(DiffImg))
    
    # Unsupervised Chabonier Photometric Loss
    # epsilon = 1e-3
    # alpha = 0.45
    # lossPhoto = tf.reduce_mean(tf.pow(tf.square(DiffImg) + tf.square(epsilon), alpha))

    return lossPhoto, WarpI1Patch
    
def TrainOperation(ImgPH, I1PH, I2PH, LabelPH, TrainNames, TestNames, NumTrainSamples, PatchSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                   DivTrain, LatestFile, LossFuncName, NetworkType, BasePath, LogsPath, TrainingType, OriginalImageSize, opt, HObj):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    HomingVecPH is the ground truth  homing vector placeholder
    DirNames - Full path to all image files without extension
    Train/Val - Idxs of all the images to be used for training/validation (held-out testing in this case)
    Train/ValLabels - Labels corresponding to Train/Val
    NumTrain/ValSamples - length(Train/Val)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of data
    LatestFile - Latest checkpointfile to continue training
    Outputs:
    Saves Trained network in CheckPointPath
    """
    
    # Predict output with forward pass
    prHVal, prVal, WarpI1Patch = ICSTN(ImgPH, PatchSize, MiniBatchSize, opt)
    # WarpI1Patch =  warp2.transformImage(opt, I1PH, prHVal)
    WarpI2PatchIdeal = warp2.transformImage(opt, I2PH, warp2.vec2mtrx(opt, LabelPH))
    
    with tf.name_scope('Loss'):
    	loss, WarpI1PatchRet = LossFunc(I1PH, I2PH, LabelPH, prHVal, prVal, MiniBatchSize, PatchSize, opt)
            
    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=OptimizerParams[0], beta1=OptimizerParams[1],
                                           beta2=OptimizerParams[2], epsilon=OptimizerParams[3])
        Gradients = Optimizer.compute_gradients(loss)
        OptimizerUpdate = Optimizer.apply_gradients(Gradients)
        #Optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(loss)
        #Optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.image('WarpI1Patch', WarpI1Patch[:,:,:,0:3])
    tf.summary.image('I1Patch', I1PH[:,:,:,0:3])
    tf.summary.image('I2Patch', I2PH[:,:,:,0:3])
    tf.summary.image('WarpI1PatchIdeal', WarpI2PatchIdeal[:,:,:,0:3])
    tf.summary.histogram('prHVal', prHVal)
    tf.summary.histogram('Label', LabelPH)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    
    AllEpochLoss = [0.0]
    EachIterLoss = [0.0]
    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Print Number of parameters in the network    
        tu.FindNumParams(1)
        
        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
            
        TotalTimeElapsed = 0.0
        TimerOverall = tic()
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            EpochLoss = 0.0
            Timer1 = tic()
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                print('Epoch ' + str(Epochs) + ' PerEpochCounter ' + str(PerEpochCounter))
                Timer2 = tic()

                IBatch, I1Batch, I2Batch, HBatch, CompositionsBatch = GenerateBatch(TrainNames, PatchSize, MiniBatchSize, HObj, BasePath, OriginalImageSize)

                FeedDict = {ImgPH: IBatch, I1PH: I1Batch, I2PH:I2Batch, LabelPH: CompositionsBatch}
                _, LossThisBatch, Summary, WarpI2PatchIdealRet = sess.run([OptimizerUpdate, loss, MergedSummaryOP, WarpI2PatchIdeal], feed_dict=FeedDict)
                
                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()

                # Calculate and print Train accuracy (also called EpochLoss) every epoch
                EpochLoss += LossThisBatch

                # Save All losses
                EachIterLoss.append(LossThisBatch)

                TimeLastMiniBatch = toc(Timer2)

                # Print LossThisBatch
                print('LossThisBatch is  '+ str(LossThisBatch))
                
                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print(SaveName + ' Model Saved...')

                # Print timing information
                EstimatedTimeToCompletionThisEpoch = float(TimeLastMiniBatch)*float(NumIterationsPerEpoch-PerEpochCounter-1.0)
                EstimatedTimeToCompletionTotal = float(TimeLastMiniBatch)*float(NumIterationsPerEpoch-PerEpochCounter-1.0) +\
                                                 float(TimeLastMiniBatch)*float(NumIterationsPerEpoch-1.0)*float(NumEpochs-Epochs)
                TotalTimeElapsed = toc(TimerOverall)
                print('Percentage complete in total epochs ' + str(float(Epochs+1)/float(NumEpochs-StartEpoch+1)*100.0))
                print('Percentage complete in this Train epoch ' + str(float(PerEpochCounter)/float(NumIterationsPerEpoch)*100.0))
                print('Last MiniBatch took '+ str(TimeLastMiniBatch) + ' secs, time taken till now ' + str(TotalTimeElapsed) + \
                      ' estimated time to completion of this epoch is ' + str(EstimatedTimeToCompletionThisEpoch))
                print('Estimated Total time remaining is ' + str(EstimatedTimeToCompletionTotal))
                
            TimeLastEpoch = toc(Timer1)
            EstimatedTimeToCompletion = float(TotalTimeElapsed)/float(Epochs+1.0)*float(NumEpochs-Epochs-1.0)
                
            # Save Each Epoch loss
            AllEpochLoss.append(EpochLoss)
            
            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print(SaveName + ' Model Saved...')


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
    Parser.add_argument('--NumEpochs', type=int, default=200, help='Number of Epochs to Train for, Default:200')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=32, help='Size of the MiniBatch to use, Default:32')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointPath?, Default:0')
    Parser.add_argument('--RemoveLogs', type=int, default=0, help='Delete log Files from ./Logs?, Default:0')
    Parser.add_argument('--LossFuncName', default='PhotoL1', help='Choice of Loss functions, choose from PhotoL1, PhotoChab, PhotoRobust. Default:PhotoL1')
    Parser.add_argument('--NetworkType', default='Large', help='Choice of Network type, choose from Small, Large, Default:Large')
    Parser.add_argument('--CheckPointPath', default='/home/nitin/PRGEye/CheckPoints/', help='Path to save checkpoints, Default:/home/nitin/PRGEye/CheckPoints/')
    Parser.add_argument('--LogsPath', default='/home/nitin/PRGEye/Logs/', help='Path to save Logs, Default:/home/nitin/PRGEye/Logs/')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--LR', type=float, default=1e-4, help='Learning Rate, Default: 1e-4')
    Parser.add_argument('--TrainingType', default='S', help='Training Type, S: Supervised, US: Unsupervised, Default: US')
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    RemoveLogs = Args.RemoveLogs
    LossFuncName = Args.LossFuncName
    NetworkType = Args.NetworkType
    CheckPointPath = Args.CheckPointPath 
    LogsPath = Args.LogsPath
    GPUDevice = Args.GPUDevice
    LearningRate = Args.LR
    TrainingType = Args.TrainingType
    
    # Set GPUDevice
    tu.SetGPU(GPUDevice)

    if(RemoveLogs is not 0):
        shutil.rmtree(os.getcwd() + os.sep + 'Logs' + os.sep)

    # Setup all needed parameters including file reading
    TrainNames, ValNames, TestNames, OptimizerParams,\
    SaveCheckPoint, PatchSize, NumTrainSamples, NumValSamples, NumTestSamples,\
    NumTestRunsPerEpoch, OriginalImageSize, HObj = SetupAll(BasePath, LearningRate)

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)

    opt = warp2.Options(PatchSize=PatchSize, MiniBatchSize=MiniBatchSize, warpType= ['pseudosimilarity', 'pseudosimilarity'])
    
    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
        
    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], 2*PatchSize[2]), name='Input')

    # PH for losses
    I1PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='I1')
    I2PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='I2')
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 3), name='Label')  

    TrainOperation(ImgPH, I1PH, I2PH, LabelPH, TrainNames, TestNames, NumTrainSamples, PatchSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                   DivTrain, LatestFile, LossFuncName, NetworkType, BasePath, LogsPath, TrainingType, OriginalImageSize, opt, HObj)
        
    
if __name__ == '__main__':
    main()

