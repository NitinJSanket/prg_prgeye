#!/usr/bin/env python

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)


# TODO: Adapt more augmentation from: https://github.com/sthalles/deeplab_v3/blob/master/preprocessing/inception_preprocessing.py
# Refer to https://github.com/chengshengchan/model_compression

import tensorflow as tf
import cv2
import sys
import os
import glob
import re
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
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
import copy
import platform

# Don't generate pyc codes
sys.dont_write_bytecode = True

@Scope
def Loss(I1PH, I2PH, C1PH, C2PH, HP1PH, HP2PH, LabelPH, prHValT, prHValS, prValT, prValS, MiniBatchSize, PatchSize, opt, Args):
    prHValT = warp2.vec2mtrx(opt, prValT)
    prHValS = warp2.vec2mtrx(opt, prValS)
    WarpI1PatchT = warp2.transformImage(opt, I1PH, prHValT)
    WarpI1PatchS = warp2.transformImage(opt, I1PH, prHValS)

    if Args.LossFuncName == 'TS':
        # NIPS 2014 Paper: https://arxiv.org/pdf/1312.6184.pdf
        Lambda = 0.0
        loss = tf.reduce_mean(tf.square(prValT - prValS))
    elif Args.LossFuncName == 'Proj':
        # NIPS 2015 Paper: https://arxiv.org/pdf/1503.02531.pdf
        # https://research.google/pubs/pub46569/
        Lambda = [1.0, 1.0, 0.1] # T, S, Proj.
        lossT = tf.square(prValT - LabelPH)
        lossS = tf.square(prValS - LabelPH)
        lossProj = tf.square(prValT - prValS)
        loss = tf.reduce_mean(Lambda[0]*lossT + Lambda[1]*lossS + Lambda[2]*lossProj)
            
    return loss, WarpI1PatchT, WarpI1PatchS, Lambda

@Scope
def Optimizer(OptimizerParams, loss):
    Optimizer = tf.train.AdamOptimizer(learning_rate=OptimizerParams[0], beta1=OptimizerParams[1],
                                           beta2=OptimizerParams[2], epsilon=OptimizerParams[3])
    Gradients = Optimizer.compute_gradients(loss)
    OptimizerUpdate = Optimizer.apply_gradients(Gradients)
    # Optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(loss)
    # Optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(loss)
    return OptimizerUpdate

def TensorBoard(loss, WarpI1PatchT, WarpI1PatchS, I1PH, I2PH, C1PH, C2PH, HP1PH, HP2PH, WarpI1PatchIdealPH, prValT, prValS, LabelPH, Args):
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.image('I1Patch', I1PH[:,:,:,0:3], max_outputs=3)
    tf.summary.image('I2Patch', I2PH[:,:,:,0:3], max_outputs=3)
    tf.summary.image('WarpI1PatchT', WarpI1PatchT[:,:,:,0:3], max_outputs=3)
    tf.summary.image('WarpI1PatchS', WarpI1PatchS[:,:,:,0:3], max_outputs=3)
    if(Args.SuperPointFlag):
         tf.summary.image('C1', C1PH[:,:,:,0:3], max_outputs=3)
         tf.summary.image('C2', C2PH[:,:,:,0:3], max_outputs=3)
    if(Args.HPFlag):
         tf.summary.image('HP1', HP1PH[:,:,:,0:3], max_outputs=3)
         tf.summary.image('HP2', HP2PH[:,:,:,0:3], max_outputs=3)
    tf.summary.histogram('prValT', prValT)
    tf.summary.histogram('prValS', prValS)
    tf.summary.histogram('Label', LabelPH)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    return MergedSummaryOP


def PrettyPrint(Args, warpType, warpTypedg, HObj, Lambda, VNT, VNS, OverideKbInput=False):
    # TODO: Write to file?
    Username = getpass.getuser()
    cprint('Running on {}'.format(Username), 'yellow')
    cprint('Network Statistics', 'yellow')
    cprint('Network Used: {}'.format(Args.NetworkName), 'yellow')
    cprint('GPU Used: {}'.format(Args.GPUDevice), 'yellow')
    cprint('Learning Rate: {}'.format(Args.LR), 'yellow')
    cprint('Teacher Model', 'yellow')
    cprint('Init Neurons {}, Expansion Factor {}, NumBlocks {}, DropOutFactor {}'.format(VNT.InitNeurons, VNT.ExpansionFactor, VNT.NumBlocks, VNT.DropOutRate), 'yellow')
    cprint('Student Model', 'yellow')
    cprint('Init Neurons {}, Expansion Factor {}, NumBlocks {}, DropOutFactor {}'.format(VNS.InitNeurons, VNS.ExpansionFactor, VNS.NumBlocks, VNS.DropOutRate), 'yellow')
    cprint('Warp Types used: {}'.format(warpType), 'green')
    cprint('Warp Types For Data Generation: {}'.format(warpTypedg), 'green')
    cprint('Loss Function used: {}'.format(Args.LossFuncName), 'green')
    cprint('Loss Function Weights: {}'.format(Lambda), 'green')
    cprint('Augmentations Used: {}'.format(Args.Augmentations), 'green')
    cprint('Input used: {}'.format(Args.Input), 'green')
    cprint('MaxParams used: {}'.format(HObj.MaxParams), 'green')
    cprint('CheckPoints are saved in: {}'.format(Args.CheckPointPath), 'red')
    cprint('Logs are saved in: {}'.format(Args.LogsPath), 'red')
    cprint('Images used for Training are in: {}'.format(Args.BasePath), 'red')
    if(OverideKbInput):
        Key = 'y'
    else:
        PythonVer = platform.python_version().split('.')[0]
        # Parse Python Version to handle super accordingly
        if (PythonVer == '2'):
            Key = raw_input('Enter y/Y/yes/Yes/YES to save to RunCommand.md, any other key to exit.')
        else:
            Key = input('Enter y/Y/yes/Yes/YES to save to RunCommand.md, any other key to exit.')
    if(Key.lower() == 'y' or Key.lower() == 'yes'):
        FileName = 'RunCommand.md'
        with open(FileName, 'a+') as RunCommand:
            RunCommand.write('\n\n')
            RunCommand.write('{}\n'.format(datetime.now()))
            RunCommand.write('Username: {}\n'.format(Username))
            RunCommand.write('Learning Rate: {}\n'.format(Args.LR))
            RunCommand.write('Network Used: {}\n'.format(Args.NetworkName))
            RunCommand.write('GPU Used: {}\n'.format(Args.GPUDevice))
            RunCommand.write('Teacher Model \n')
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, DropOutFactor {}\n'.format(VNT.InitNeurons, VNT.ExpansionFactor, VNT.NumBlocks, VNT.DropOutRate))
            RunCommand.write('Student Model \n')
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, DropOutFactor {}\n'.format(VNS.InitNeurons, VNS.ExpansionFactor, VNS.NumBlocks, VNS.DropOutRate))
            RunCommand.write('Warp Types used: {}\n'.format(warpType))
            RunCommand.write('Warp Types For Data Generation: {}\n'.format(warpTypedg))
            RunCommand.write('Loss Function used: {}\n'.format(Args.LossFuncName))
            RunCommand.write('Loss Function Weights: {}\n'.format(Lambda))
            RunCommand.write('Augmentations Used: {}\n'.format(Args.Augmentations))
            RunCommand.write('Input used: {}\n'.format(Args.Input))
            RunCommand.write('MaxParams used: {}\n'.format(HObj.MaxParams))
            RunCommand.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
            RunCommand.write('Logs are saved in: {}\n'.format(Args.LogsPath))
            RunCommand.write('Images used for Training are in: {}\n'.format(Args.BasePath))
        cprint('Log written in {}'.format(FileName), 'yellow')
        FileName = Args.CheckPointPath + 'RunCommand.md'
        with open(FileName, 'w+') as RunCommand:
            RunCommand.write('\n\n')
            RunCommand.write('{}\n'.format(datetime.now()))
            RunCommand.write('Username: {}\n'.format(Username))
            RunCommand.write('Learning Rate: {}\n'.format(Args.LR))
            RunCommand.write('Network Used: {}\n'.format(Args.NetworkName))
            RunCommand.write('GPU Used: {}\n'.format(Args.GPUDevice))
            RunCommand.write('Teacher Model \n')
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, DropOutFactor {}\n'.format(VNT.InitNeurons, VNT.ExpansionFactor, VNT.NumBlocks, VNT.DropOutRate))
            RunCommand.write('Student Model \n')
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, DropOutFactor {}\n'.format(VNS.InitNeurons, VNS.ExpansionFactor, VNS.NumBlocks, VNS.DropOutRate))
            RunCommand.write('Warp Types used: {}\n'.format(warpType))
            RunCommand.write('Warp Types For Data Generation: {}\n'.format(warpTypedg))
            RunCommand.write('Loss Function used: {}\n'.format(Args.LossFuncName))
            RunCommand.write('Loss Function Weights: {}\n'.format(Lambda))
            RunCommand.write('Augmentations Used: {}\n'.format(Args.Augmentations))
            RunCommand.write('Input used: {}\n'.format(Args.Input))
            RunCommand.write('MaxParams used: {}\n'.format(HObj.MaxParams))
            RunCommand.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
            RunCommand.write('Logs are saved in: {}\n'.format(Args.LogsPath))
            RunCommand.write('Images used for Training are in: {}\n'.format(Args.BasePath))
        cprint('Log written in {}'.format(FileName), 'yellow')
    else:
        cprint('Log writing skipped', 'yellow')
        
    
def TrainOperation(ImgPH, I1PH, I2PH, C1PH, C2PH, HP1PH, HP2PH, LabelPH, IOrgPH, HPH, WarpI1PatchIdealPH, TrainNames, TestNames, NumTrainSamples, PatchSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                   DivTrain, LatestFile, LossFuncName, NetworkType, BasePath, LogsPath, OriginalImageSize, opt, optdg, HObj, Args, warpType):
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
    # Teacher
    # Create Network Object with required parameters
    # Import Network Module
    NetT = importlib.import_module(Args.NetworkName)
    ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
    Network = getattr(NetT, ClassName)
    VNT = Network(InputPH = ImgPH, Training = True, Opt = opt, InitNeurons = Args.InitNeuronsT, Suffix = 'Teacher')
    # Predict output with forward pass
    # WarpI1Patch contains warp of both I1 and I2, extract first three channels for useful data
    prHValT, prValT, _ = VNT.Network()

    # Student
    NetS = importlib.import_module(Args.NetworkName + 'Small')
    # Currently Only supports same type models
    Network = getattr(NetS, ClassName)
    VNS = Network(InputPH = ImgPH, Training = True, Opt = opt, InitNeurons = Args.InitNeuronsS, Suffix = 'Student')
    # Predict output with forward pass
    # WarpI1Patch contains warp of both I1 and I2, extract first three channels for useful data
    prHValS, prValS, _ = VNS.Network()
    
    # Warp I1 with ideal parameters for visual sanity check
    # MODIFY THIS DEPENDING ON ARCH!
    opt2 = copy.copy(opt)
    opt2.warpType = 'pseudosimilarity'
    WarpI1PatchIdeal = warp2.transformImage(opt, IOrgPH, warp2.vec2mtrx(opt2, LabelPH))

    # Data Generation
    # MODIFY THIS DEPENDING ON ARCH!
    optdg.warpType = 'pseudosimilarity'
    # HObj.TranformType is set in DataHandling.py
    I2Gen = warp2.transformImage(optdg, IOrgPH, HPH)

    # Compute Loss
    loss, WarpI1PatchT, WarpI1PatchS, Lambda = Loss(I1PH, I2PH, C1PH, C2PH, HP1PH, HP2PH, LabelPH, prHValT, prHValS, prValT, prValS, MiniBatchSize, PatchSize, opt2, Args)

    # Run Backprop and Gradient Update
    OptimizerUpdate = Optimizer(OptimizerParams, loss)
        
    # Tensorboard
    
    MergedSummaryOP = TensorBoard(loss, WarpI1PatchT, WarpI1PatchS, I1PH, I2PH, C1PH, C2PH, HP1PH, HP2PH, WarpI1PatchIdealPH, prValT, prValS, LabelPH, Args)

    AllVars = tf.global_variables()
    VarsT = [Vars for Vars in AllVars if 'Teacher' in Vars.name]
    VarsS = [Vars for Vars in AllVars if 'Student' in Vars.name]
        
    # Setup Saver
    SaverT = tf.train.Saver(VarsT)
    SaverS = tf.train.Saver(VarsS)

    try:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if Args.DistillType == 'TS':
                # Load Teacher Model
                SaverT.restore(sess, Args.TeacherCheckPoint)
            if LatestFile is not None:
                SaverS.restore(sess, CheckPointPath + LatestFile + '.ckpt')
                # Extract only numbers from the name
                StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit())) + 1
                print('Loaded latest checkpoint with the name ' + LatestFile + '....')
            else:
                StartEpoch = 0
                print('New model initialized....')

            # Create Batch Generator Object
            bg = BatchGeneration(sess, I2Gen, IOrgPH, HPH, SuperPointFlag = Args.SuperPointFlag)

            # Create Data Augmentation Object
            if(Args.DataAug):
                Args.Augmentations =  ['Brightness', 'Contrast', 'Hue', 'Saturation', 'Gamma', 'Gaussian']
                da = iu.DataAugmentationTF(sess, I1PH, Augmentations = Args.Augmentations)
                DataAugGen = da.RandPerturbBatch()
            else:
                Args.Augmentations = 'None'
                DataAugGen = None
                da = None

            # Pretty Print Stats
            PrettyPrint(Args, warpType, opt2.warpType, HObj, Lambda, VNT, VNS, OverideKbInput=False)

            # Tensorboard
            Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

            for Epochs in tqdm(range(StartEpoch, NumEpochs)):
                NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
                for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                    IBatch, I1Batch, I2Batch, P1Batch, P2Batch, C1Batch, C2Batch, HBatch, ParamsBatch =\
                        bg.GenerateBatchTF(TrainNames, PatchSize, MiniBatchSize, HObj, BasePath, OriginalImageSize, Args, da, DataAugGen)

                    # Parse for loss functions on different inputs
                    if 'HP' in Args.LossFuncName:
                        try:
                            P1Batch = iu.HPFilterBatch(P1Batch)
                            P2Batch = iu.HPFilterBatch(P2Batch)
                        except:
                            pass
                    elif 'SP' in Args.LossFuncName:
                        P1Batch = C1Batch
                        P2Batch = C2Batch
                    elif 'G' in Args.LossFuncName:
                        try:
                            P1Batch = np.tile(iu.rgb2gray(P1Batch), (1,1,1,3))
                            P2Batch = np.tile(iu.rgb2gray(P2Batch), (1,1,1,3))
                        except:
                            pass
                    if Args.SuperPointFlag:
                        FeedDict = {VN.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, LabelPH: ParamsBatch, IOrgPH: I1Batch, C1PH: C1Batch, C2PH:C2Batch}
                        _, LossThisBatch, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)
                    elif Args.HPFlag:
                        HP1Batch = iu.HPFilterBatch(P1Batch)
                        HP2Batch = iu.HPFilterBatch(P2Batch)
                        FeedDict = {VN.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, LabelPH: ParamsBatch, IOrgPH: I1Batch, HP1PH: HP1Batch, HP2PH:HP2Batch}
                        _, LossThisBatch, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)
                    elif Args.HPFlag and Args.SuperPointFlag:
                        HP1Batch = iu.HPFilterBatch(P1Batch)
                        HP2Batch = iu.HPFilterBatch(P2Batch)
                        FeedDict = {VN.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, LabelPH: ParamsBatch, IOrgPH: I1Batch, C1PH: C1Batch, C2PH:C2Batch,\
                         HP1PH: HP1Batch, HP2PH:HP2Batch}
                        _, LossThisBatch, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)
                    else:
                        if(Args.LossFuncName == 'TS'):
                            FeedDict = {VNT.InputPH: IBatch}
                            prValTRet = sess.run([prValT], feed_dict=FeedDict)[0]
                            FeedDict = {VNS.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, LabelPH: prValTRet, IOrgPH: I1Batch}
                            _, LossThisBatch, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)
                            print('ERROR: Not Implemented Yet!')
                            sys.exit()
                        else:
                            FeedDict = {VNT.InputPH: IBatch, VNS.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, LabelPH: ParamsBatch, IOrgPH: I1Batch}
                            _, LossThisBatch, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)
                   

                    # Tensorboard
                    Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                    # If you don't flush the tensorboard doesn't update until a lot of iterations!
                    Writer.flush()

                    # Save checkpoint every some SaveCheckPoint's iterations
                    if PerEpochCounter % SaveCheckPoint == 0:
                        # Save the Model learnt in this epoch
                        SaveNameT =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'modelT.ckpt'
                        SaverT.save(sess,  save_path=SaveNameT)
                        print(SaveNameT + ' Model Saved...')
                        SaveNameS =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'modelS.ckpt'
                        SaverS.save(sess,  save_path=SaveNameS)
                        print(SaveNameS + ' Model Saved...')

                # Save model every epoch
                SaveNameT = CheckPointPath + str(Epochs) + 'modelT.ckpt'
                SaverT.save(sess, save_path=SaveNameT)
                print(SaveNameT + ' Model Saved...')
                SaveNameS = CheckPointPath + str(Epochs) + 'modelS.ckpt'
                SaverS.save(sess, save_path=SaveNameS)
                print(SaveNameS + ' Model Saved...')

        # Pretty Print Stats before exiting
        PrettyPrint(Args, warpType, opt2.warpType, HObj, Lambda, VNT, VNS, OverideKbInput=True)
     
    except KeyboardInterrupt:
        # Pretty Print Stats before exitting
        PrettyPrint(Args, warpType, opt2.warpType, HObj, Lambda, VNT, VNS)

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
    Parser.add_argument('--NetworkName', default='Network.VanillaNet', help='Name of network file, Default: Network.ResNet')
    Parser.add_argument('--CheckPointPath', default='/home/nitin/PRGEye/CheckPoints/', help='Path to save checkpoints, Default:/home/nitin/PRGEye/CheckPoints/')
    Parser.add_argument('--LogsPath', default='/home/nitin/PRGEye/Logs/', help='Path to save Logs, Default:/home/nitin/PRGEye/Logs/')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--DataAug', type=int, default=0, help='Do you want to do Data augmentation?, Default:0')
    Parser.add_argument('--LR', type=float, default=1e-4, help='Learning Rate, Default: 1e-4')
    Parser.add_argument('--InitNeuronsT', type=float, default=36, help='Starting Number of Neurons in Teacher, Default: 36')
    Parser.add_argument('--InitNeuronsS', type=float, default=36, help='Starting Number of Neurons in Student, Default: 20')
    Parser.add_argument('--Input', default='I', help='Input, choose from I: RGB Images, G: Grayscale Images, HP: HP Grayscale Images, SP: Cornerness, Default: I')
    Parser.add_argument('--TeacherCheckPoint', default='/home/nitin/Models/model.ckpt', help='Teacher Model Weights, used for TS Mode, Default: /home/nitin/Models/model.ckpt')
    Parser.add_argument('--DistillType', default='TS', help='Distillation Method, choose from TS: Teacher Student, P: Projection Like Loss, Default: TS')
    
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
    Args.SuperPointFlag = ('SP' in Args.LossFuncName) or ('SP' in Args.RegFuncName) or ('PhotoRobust' in Args.LossFuncName) or('PhotoRobust' in Args.RegFuncName)
    Args.HPFlag = ('HP' in Args.LossFuncName) or ('HP' in Args.RegFuncName)

    # Set GPUDevice
    tu.SetGPU(GPUDevice)


    if(RemoveLogs is not 0):
        shutil.rmtree(os.getcwd() + os.sep + 'Logs' + os.sep)

    # Setup all needed parameters including file reading
    # MODIFY THIS DEPENDING ON ARCHITECTURE!
    warpType = ['pseudosimilarity'] # ['translation', 'translation', 'scale', 'scale'] # ['pseudosimilarity', 'pseudosimilarity'] # ['translation', 'translation', 'scale', 'scale']  # ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity'] #, 'pseudosimilarity']#, 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity'] # ['translation', 'translation', 'scale', 'scale'] 
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
    C1PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='C1')
    C2PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='C2')
    HP1PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='HP1')
    HP2PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='HP2')
    WarpI1PatchIdealPH =  tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='I1WarpIdeal')
    # MODIFY THIS DEPENDING ON ARCH!
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 3), name='Label')
    # PH for Data Generation
    IOrgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, OriginalImageSize[0], OriginalImageSize[1], OriginalImageSize[2]), name='IOrg') 
    HPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 3, 3), name='LabelH')

    TrainOperation(ImgPH, I1PH, I2PH, C1PH, C2PH, HP1PH, HP2PH, LabelPH, IOrgPH, HPH, WarpI1PatchIdealPH, TrainNames, TestNames, NumTrainSamples, PatchSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch,
                       DivTrain, LatestFile, LossFuncName, NetworkType, BasePath, LogsPath, OriginalImageSize, opt, optdg, HObj, Args, warpType)
    
    
if __name__ == '__main__':
    main()

