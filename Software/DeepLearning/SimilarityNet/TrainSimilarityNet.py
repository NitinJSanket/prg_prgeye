#!/usr/bin/env python

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)


# TODO: Adapt more augmentation from: https://github.com/sthalles/deeplab_v3/blob/master/preprocessing/inception_preprocessing.py

import tensorflow as tf
import cv2
import sys
import os
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
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
from Misc.Decorators import *
# Import of network is done in main code
import importlib
from datetime import datetime
import getpass

# Don't generate pyc codes
sys.dont_write_bytecode = True

@Scope
def Loss(I1PH, I2PH, LabelPH, prHVal, prVal, MiniBatchSize, PatchSize, opt, Args):
    WarpI1Patch = warp2.transformImage(opt, I1PH, prHVal)
    Lambda = [1.0, 10.0, 10.0]
    LambdaStack = np.tile(Lambda, (MiniBatchSize, 1))

    # Choice of Loss Function
    if(Args.LossFuncName == 'SL2'):
        # Supervised L2 loss
        lossPhoto = tf.reduce_mean(tf.square(tf.multiply(prVal - LabelPH, LambdaStack)))
    elif(Args.LossFuncName == 'PhotoL1'):        
        # Self-supervised Photometric L1 Losses
        DiffImg = WarpI1Patch - I2PH

        # Self-supervised Photometric L1 Loss
        lossPhoto = tf.reduce_mean(tf.abs(tf.multiply(DiffImg, LambdaStack)))
    elif(Args.LossFuncName == 'PhotoChab'):
        # Self-supervised Photometric Chabonier Loss
        epsilon = 1e-3
        alpha = 0.45
        lossPhoto = tf.reduce_mean(tf.pow(tf.square(tf.multiply(DiffImg, LambdaStack)) + tf.square(epsilon), alpha))
    elif(Args.LossFuncName == 'PhotoRobust'):
        print('ERROR: Not implemented yet!')
        sys.exit(0)

    if(Args.RegFuncName == 'None'):
        lossReg = 0.
    elif(Args.RegFuncName == 'C'):
        # TODO: Cornerness Loss
        # WarpI1Patch = tf.boolean_mask(WarpI1, MaskPH)
        # WarpI1PatchCornerness = tf.boolean_mask(WarpI1Cornerness, MaskPH), send this as input to function
        # I2PatchCornerness = tf.boolean_mask(I2CornernessPH, MaskPH), send this as input to function
        print('ERROR: Not implemented yet!')
        sys.exit(0)
    
    # TODO: HP Filter Loss
    # Lambda = [0.1, 1.0] # Photo, CornerPhoto
    # lossCornerPhoto = tf.math.multiply(WarpI1Patch, WarpI1PatchCornerness) - tf.math.multiply(I2Patch, I2PatchCornerness)
    # lossPhoto = tf.reduce_mean(Lambda[0]*tf.abs(DiffImg) + Lambda[1]*lossCornerPhoto)
    
    # loss = lossPhoto + lossReg

    return lossPhoto, WarpI1Patch, Lambda

@Scope
def Optimizer(OptimizerParams, loss):
    Optimizer = tf.train.AdamOptimizer(learning_rate=OptimizerParams[0], beta1=OptimizerParams[1],
                                           beta2=OptimizerParams[2], epsilon=OptimizerParams[3])
    Gradients = Optimizer.compute_gradients(loss)
    OptimizerUpdate = Optimizer.apply_gradients(Gradients)
    # Optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(loss)
    # Optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(loss)
    return OptimizerUpdate

def TensorBoard(loss, WarpI1Patch, I1PH, I2PH, WarpI1PatchIdealPH, prVal, LabelPH):
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.image('I1Patch', I1PH[:,:,:,0:3], max_outputs=3)
    tf.summary.image('I2Patch', I2PH[:,:,:,0:3], max_outputs=3)
    tf.summary.image('WarpI1Patch', WarpI1Patch[:,:,:,0:3], max_outputs=3)
    I2PHGray = tf.image.rgb_to_grayscale(I2PH)
    WarpI1PatchGray = tf.image.rgb_to_grayscale(WarpI1Patch[:,:,:,0:3]*255.0)
    OverlayImg = tf.concat([tf.concat([I2PHGray, tf.zeros(np.shape(I2PHGray))], axis=3), WarpI1PatchGray], axis=3)
    tf.summary.image('DiffOverlay', OverlayImg, max_outputs=3)
    # tf.summary.image('WarpI1PatchIdeal', WarpI1PatchIdealPH[:,:,:,0:3], max_outputs=3)
    tf.summary.histogram('prVal', prVal)
    tf.summary.histogram('Label', LabelPH)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    return MergedSummaryOP


def PrettyPrint(Args, NumParams, NumFlops, ModelSize, warpType, warpTypedg, Lambda, VN, OverideKbInput=False):
    # TODO: Write to file?
    Username = getpass.getuser()
    cprint('Running on {}'.format(Username), 'yellow')
    cprint('Network Statistics', 'yellow')
    cprint('Network Used: {}'.format(Args.NetworkName), 'yellow')
    cprint('Learning Rate: {}'.format(Args.LR), 'yellow')
    cprint('Init Neurons {}, Expansion Factor {}, NumBlocks {}, DropOutFactor {}'.format(VN.InitNeurons, VN.ExpansionFactor, VN.NumBlocks, VN.DropOutRate), 'yellow')
    cprint('Num Params: {}'.format(NumParams), 'green')
    cprint('Num FLOPs: {}'.format(NumFlops), 'green')
    cprint('Estimated Model Size (MB): {}'.format(ModelSize), 'green')
    cprint('Warp Types used: {}'.format(warpType), 'green')
    cprint('Warp Types For Data Generation: {}'.format(warpTypedg), 'green')
    cprint('Loss Function used: {}'.format(Args.LossFuncName), 'green')
    cprint('Loss Function Weights: {}'.format(Lambda), 'green')
    cprint('CheckPoints are saved in: {}'.format(Args.CheckPointPath), 'red')
    cprint('Logs are saved in: {}'.format(Args.LogsPath), 'red')
    cprint('Images used for Training are in: {}'.format(Args.BasePath), 'red')
    if(OverideKbInput):
        Key = 'y'
    else:
        Key = raw_input('Enter y/Y/yes/Yes/YES to save to RunCommand.md, any other key to exit.')
    if(Key.lower() == 'y' or Key.lower() == 'yes'):
        FileName = 'RunCommand.md'
        with open(FileName, 'a+') as RunCommand:
            RunCommand.write('\n\n')
            RunCommand.write('{}\n'.format(datetime.now()))
            RunCommand.write('Username: {}\n'.format(Username))
            RunCommand.write('Learning Rate: {}\n'.format(Args.LR))
            RunCommand.write('Network Used: {}\n'.format(Args.NetworkName))
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, DropOutFactor {}\n'.format(VN.InitNeurons, VN.ExpansionFactor, VN.NumBlocks, VN.DropOutRate))
            RunCommand.write('Num Params: {}\n'.format(NumParams))
            RunCommand.write('Num FLOPs: {}\n'.format(NumFlops))
            RunCommand.write('Estimated Model Size (MB): {}\n'.format(ModelSize))
            RunCommand.write('Warp Types used: {}\n'.format(warpType))
            RunCommand.write('Warp Types For Data Generation: {}\n'.format(warpTypedg))
            RunCommand.write('Loss Function used: {}\n'.format(Args.LossFuncName))
            RunCommand.write('Loss Function Weights: {}\n'.format(Lambda))
            RunCommand.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
            RunCommand.write('Logs are saved in: {}\n'.format(Args.LogsPath))
            RunCommand.write('Images used for Training are in: {}\n'.format(Args.BasePath))
        cprint('Log written in {}'.format(FileName), 'yellow')
    else:
        cprint('Log writing skipped', 'yellow')
        
    
def TrainOperation(ImgPH, I1PH, I2PH, LabelPH, IOrgPH, HPH, WarpI1PatchIdealPH, TrainNames, TestNames, NumTrainSamples, PatchSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                   DivTrain, LatestFile, LossFuncName, NetworkType, BasePath, LogsPath, OriginalImageSize, opt, optdg, HObj, Net, Args, warpType, InitNeurons):
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
    # Create Network Object with required parameters
    VN = Net.VanillaNet(InputPH = ImgPH, Training = True, Opt = opt, InitNeurons = InitNeurons)
    # Predict output with forward pass
    # WarpI1Patch contains warp of both I1 and I2, extract first three channels for useful data
    prHVal, prVal, WarpI1Patch = VN.Network()

    # TODO: Warp Patch here
    # Maybe Asmall * AbigInv * H * Abig
    # Warp I1 with ideal parameters for visual sanity check
    # MODIFY THIS DEPENDING ON ARCH!
    opt2 = opt
    opt2.warpType = 'pseudosimilarity'
    # optlarge = warp2.Options(PatchSize=OriginalImageSize, MiniBatchSize=MiniBatchSize, warpType = 'pseudosimilarity') # ICSTN Options
    # Alarge = tf.linalg.inv(optlarge.refMtrx)
    # HCorr = tf.matmul(Alarge, tf.matmul(warp2.vec2mtrx(opt2, LabelPH), tf.linalg.inv(Alarge)))
    # WarpI1PatchIdeal = warp2.transformImage(opt, I1PH, HCorr)
    WarpI1PatchIdeal = warp2.transformImage(opt, IOrgPH, warp2.vec2mtrx(opt2, LabelPH))

    # Data Generation
    # MODIFY THIS DEPENDING ON ARCH!
    optdg.warpType = 'pseudosimilarity'
    # HObj.TranformType is set in DataHandling.py
    I2Gen = warp2.transformImage(optdg, IOrgPH, HPH)

    # Compute Loss
    loss, WarpI1PatchRet, Lambda = Loss(I1PH, I2PH, LabelPH, prHVal, prVal, MiniBatchSize, PatchSize, opt, Args)

    # Run Backprop and Gradient Update
    OptimizerUpdate = Optimizer(OptimizerParams, loss)
        
    # Tensorboard
    MergedSummaryOP = TensorBoard(loss, WarpI1Patch, I1PH, I2PH, WarpI1PatchIdealPH, prVal, LabelPH)
 
    # Setup Saver
    Saver = tf.train.Saver()

    try:
        with tf.Session() as sess:       
            if LatestFile is not None:
                Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
                # Extract only numbers from the name
                StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit())) + 1
                print('Loaded latest checkpoint with the name ' + LatestFile + '....')
            else:
                sess.run(tf.global_variables_initializer())
                StartEpoch = 0
                print('New model initialized....')

            # Create Batch Generator Object
            bg = BatchGeneration(sess, I2Gen, IOrgPH, HPH)

            # Print out Number of parameters
            NumParams = tu.FindNumParams(1)
            # Print out Number of Flops
            NumFlops = tu.FindNumFlops(sess, 1)
            # Print out Expected Model Size
            ModelSize = tu.CalculateModelSize(1)

            # Pretty Print Stats
            PrettyPrint(Args, NumParams, NumFlops, ModelSize, warpType, opt2.warpType, Lambda, VN, OverideKbInput=False)

            # Tensorboard
            Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

            for Epochs in tqdm(range(StartEpoch, NumEpochs)):
                NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
                for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                    IBatch, I1Batch, I2Batch, P1Batch, P2Batch, HBatch, ParamsBatch = bg.GenerateBatchTF(TrainNames, PatchSize, MiniBatchSize, HObj, BasePath, OriginalImageSize)
                    # P1BatchPad = iu.PadOutside(P1Batch, OriginalImageSize)

                    FeedDict = {VN.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, LabelPH: ParamsBatch, IOrgPH: I1Batch}
                    _, LossThisBatch, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)
                    # _, LossThisBatch, Summary, WarpI1PatchIdealRet = sess.run([OptimizerUpdate, loss, MergedSummaryOP, WarpI1PatchIdeal], feed_dict=FeedDict)
                    # WarpI1PatchIdealRet = iu.CenterCrop(WarpI1PatchIdealRet, PatchSize)
                    # FeedDict = {WarpI1PatchIdealPH: WarpI1PatchIdealRet, VN.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, LabelPH: ParamsBatch, IOrgPH: P1BatchPad}
                    # Summary = sess.run([MergedSummaryOP], feed_dict=FeedDict)

                    # A = np.uint8(np.concatenate((P1Batch[0], P2Batch[0], WarpI1PatchIdealRet[0], np.abs(P2Batch[0]-WarpI1PatchIdealRet[0])), axis=1))
                    # B = np.uint8(np.concatenate((I1Batch[0], I2Batch[0]), axis=1))
                    # cv2.imshow('P1, P2, P1Warp', A)
                    # cv2.imshow('I1, I2', B)
                    # cv2.waitKey(0)

                    # Tensorboard
                    Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                    # If you don't flush the tensorboard doesn't update until a lot of iterations!
                    Writer.flush()

                    # Save checkpoint every some SaveCheckPoint's iterations
                    if PerEpochCounter % SaveCheckPoint == 0:
                        # Save the Model learnt in this epoch
                        SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                        Saver.save(sess,  save_path=SaveName)
                        print(SaveName + ' Model Saved...')

                # Save model every epoch
                SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
                Saver.save(sess, save_path=SaveName)
                print(SaveName + ' Model Saved...')

        # Pretty Print Stats before exiting
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, warpType, opt2.warpType, Lambda, VN, OverideKbInput=True)
    
    except KeyboardInterrupt:
        # Pretty Print Stats before exitting
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, warpType, opt2.warpType, Lambda)

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/nitin/Datasets/MSCOCO/train2014Processed', help='Base path of images, Default:/home/nitin/Datasets/MSCOCO/train2014')
    Parser.add_argument('--NumEpochs', type=int, default=100, help='Number of Epochs to Train for, Default:200')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=32, help='Size of the MiniBatch to use, Default:32')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointPath?, Default:0')
    Parser.add_argument('--RemoveLogs', type=int, default=0, help='Delete log Files from ./Logs?, Default:0')
    Parser.add_argument('--LossFuncName', default='SL2', help='Choice of Loss functions, choose from SL2, PhotoL1, PhotoChab, PhotoRobust. Default:SL2')
    Parser.add_argument('--RegFuncName', default='None', help='Choice of regularization function, choose from None, C (Cornerness). Default:None')
    Parser.add_argument('--NetworkType', default='Large', help='Choice of Network type, choose from Small, Large, Default:Large')
    Parser.add_argument('--NetworkName', default='Network.VanillaNet3', help='Name of network file, Default: Network.VanillaNet2')
    Parser.add_argument('--CheckPointPath', default='/home/nitin/PRGEye/CheckPoints/', help='Path to save checkpoints, Default:/home/nitin/PRGEye/CheckPoints/')
    Parser.add_argument('--LogsPath', default='/home/nitin/PRGEye/Logs/', help='Path to save Logs, Default:/home/nitin/PRGEye/Logs/')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--DataAug', type=int, default=0, help='Do you want to do Data augmentation?, Default:0')
    Parser.add_argument('--LR', type=float, default=1e-4, help='Learning Rate, Default: 1e-4')
    Parser.add_argument('--InitNeurons', type=float, default=8, help='Learning Rate, Default: 8')
    
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
    NetworkName = Args.NetworkName
    DataAug = Args.DataAug
    RegFuncName = Args.RegFuncName

    # Import Network Module
    Net = importlib.import_module(NetworkName)

    # Set GPUDevice
    tu.SetGPU(GPUDevice)

    if(RemoveLogs is not 0):
        shutil.rmtree(os.getcwd() + os.sep + 'Logs' + os.sep)

    # Setup all needed parameters including file reading
    # MODIFY THIS DEPENDING ON ARCHITECTURE!
    InitNeurons = Args.InitNeurons
    warpType = ['translation', 'translation', 'scale', 'scale']  # ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity'] #, 'pseudosimilarity']#, 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity'] # ['translation', 'translation', 'scale', 'scale'] 
    TrainNames, ValNames, TestNames, OptimizerParams,\
    SaveCheckPoint, PatchSize, NumTrainSamples, NumValSamples, NumTestSamples,\
    NumTestRunsPerEpoch, OriginalImageSize, HObj, warpType = SetupAll(BasePath, LearningRate, MiniBatchSize, warpType =  warpType)

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)

    opt = warp2.Options(PatchSize=PatchSize, MiniBatchSize=MiniBatchSize, warpType = warpType) # ICSTN Options
    # opt = warp2.Options(PatchSize=OriginalImageSize, MiniBatchSize=MiniBatchSize, warpType = warpType) # ICSTN Options
    # Data Generation Options, warpType should the same the last one in the previous command
    optdg = warp2.Options(PatchSize=OriginalImageSize, MiniBatchSize=MiniBatchSize, warpType = [warpType[-1]]) 
    
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
    WarpI1PatchIdealPH =  tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='I1WarpIdeal')
    # MODIFY THIS DEPENDING ON ARCH!
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 3), name='Label')
    # PH for Data Generation
    IOrgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, OriginalImageSize[0], OriginalImageSize[1], OriginalImageSize[2]), name='IOrg') 
    HPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 3, 3), name='LabelH')

    TrainOperation(ImgPH, I1PH, I2PH, LabelPH, IOrgPH, HPH, WarpI1PatchIdealPH, TrainNames, TestNames, NumTrainSamples, PatchSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                       DivTrain, LatestFile, LossFuncName, NetworkType, BasePath, LogsPath, OriginalImageSize, opt, optdg, HObj, Net, Args, warpType, InitNeurons)
    
    
if __name__ == '__main__':
    main()

