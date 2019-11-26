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
from Network.HomographyNetICSTN import  ICSTN
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
    TrainPath = ReadPath + os.sep + 'Train.txt'
    DirNames, TrainNames = ReadDirNames(DirNamesPath, TrainPath)
    
    # Image Input Shape
    PatchSize = np.array([128, 128, 3])
    ImageSize = np.array([300, 300, 3])
    NumTrainSamples = len(TrainNames)
    
    return TrainNames, ImageSize, PatchSize, NumTrainSamples

def GenerateRandPatch(I, Rho, PatchSize, CropType, ImageSize=None, Vis=False):
    """
    Inputs: 
    I is the input image
    Vis when enabled, Visualizes the image and the perturbed image 
    Outputs:
    IPatch
    Points are labeled as:
    
    Top Left = p1, Top Right = p2, Bottom Right = p3, Bottom Left = p4 (Clockwise from Top Left)
    Code adapted from: https://github.com/mez/deep_homography_estimation/blob/master/Dataset_Generation_Visualization.ipynb
    """

    if(ImageSize is None):
        ImageSize = np.shape(I) 
    
    CenterX = PatchSize[1]/2
    CenterY = PatchSize[0]/2
    if(CropType == 'C'):
        RandX = int(np.floor(ImageSize[1]/2 - PatchSize[1]/2))
        RandY = int(np.floor(ImageSize[0]/2 - PatchSize[0]/2))
    elif(CropType == 'R'):
         RandX = int(random.randint(0, ImageSize[1]-PatchSize[1]))
         RandY = int(random.randint(0, ImageSize[0]-PatchSize[0]))
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
        cv2.polylines(IDisp, np.int32([AllPts]), 1, (255,255,255))
        cv2.imshow('a', IDisp)
        cv2.waitKey(1)

    PerturbPts = []
    for point in AllPts:
        if(len(Rho) == 1):
            # If only 1 value of Rho is given, perturb by [-Rho, Rho]
            PerturbPts.append((point[0] + random.randint(-Rho[0],Rho[0]), point[1] + random.randint(-Rho[0],Rho[0])))
        elif(len(Rho) == 2):
            if(Rho[0] != Rho[1]):
                # If bounds on Rho are given, perturb by a random value in [Rho1, Rho2] union [-Rho2, -Rho1] if Rho2 > Rho1
                PerturbPts.append((point[0] + random.choice(range(Rho[0], Rho[1]+1))*random.choice([-1, 1]),\
                                   point[1] + random.choice(range(Rho[0], Rho[1]+1))*random.choice([-1, 1])))
            else:
                # If Rho1 = Rho2 (Perturb with that amount)
                PerturbPts.append((point[0] + Rho[0], point[1] + Rho[0]))

    if(Vis is True):
        PertubImgDisp = I.copy()
        cv2.polylines(PertubImgDisp, np.int32([PerturbPts]), 1, (255,255,255))
        cv2.imshow('b', PertubImgDisp)
        cv2.waitKey(1)
        
    # Obtain Homography between the 2 images
    H = cv2.getPerspectiveTransform(np.float32(AllPts), np.float32(PerturbPts))
    # Get Inverse Homography
    HInv = np.linalg.inv(H)

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

    # I is the Original Image I1
    # WarpedI is I2
    # CroppedI is I1 cropped to patch Size
    # CroppedWarpeI is I1 Crop Warped (I2 Crop)
    # AllPts is the patch corners in I1
    # PerturbPts is the patch corners of I2 in I1
    # H8El is the first 8 elements of the Homography matrix from AllPts to PerturbPts
    # Mask is the active region of I1Patch in I1
    return I, WarpedI, CroppedI, CroppedWarpedI, AllPts, PerturbPts, H8El, Mask, H


def ReadDirNames(DirNamesPath, TrainPath):
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

    # Read TestIdxs file
    TrainIdxs = open(TrainPath, 'r')
    TrainIdxs = TrainIdxs.read()
    TrainIdxs = TrainIdxs.split()
    TrainIdxs = [int(val) for val in TrainIdxs]
    TrainNames = [DirNames[i] for i in TrainIdxs]

    return DirNames, TrainNames

def GenerateBatch(IBuffer, Rho, PatchSize, CropType, Vis=False):
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
    IBatch = []
    I1Batch = []
    I2Batch = []
    AllPtsBatch = []
    PerturbPtsBatch = []
    MaskBatch = []
    HBatch = []

    # Generate random image
    I1 = IBuffer

    # Homography and Patch generation
    I1Original, I2Original, I1Patch, I2Patch, AllPts, PerturbPts,\
    H8El, Mask, H = GenerateRandPatch(I1, Rho, PatchSize, CropType, Vis=Vis) # Rand Patch will take the whole image as it doesn't have a choice
    ICombo = np.dstack((I1Patch, I2Patch))
    
    # Normalize Dataset
    # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
    IS = iu.StandardizeInputs(np.float32(ICombo))
    
    # Append All Images and Mask
    IBatch.append(IS)
    I1Batch.append(I1Patch)
    I2Batch.append(I2Patch)
    AllPtsBatch.append(AllPts)
    PerturbPtsBatch.append(PerturbPts)
    HBatch.append(H)
    MaskBatch.append(MaskBatch)

    # IBatch is the Original Image I1 Batch
    return IBatch, I1Batch, I2Batch, AllPtsBatch, PerturbPtsBatch, HBatch, MaskBatch

            
def TestOperation(PatchPH, I1PH, I2PH, prHTruePH, PatchSize, ModelPath, ReadPath, WritePath, TrainNames, NumTrainSamples, CropType, MiniBatchSize, opt):
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
    ImageFormat = '.jpg'
    Prefix = 'COCO_train2014_%012d'
    
    # Generate indexes for center crop of train size
    CenterX = PatchSize[1]/2
    CenterY = PatchSize[0]/2
    RandX = np.ceil(CenterX - PatchSize[1]/2)
    RandY = np.ceil(CenterY - PatchSize[0]/2)
    p1 = (RandX, RandY)
    p2 = (RandX, RandY + PatchSize[0])
    p3 = (RandX + PatchSize[1], RandY + PatchSize[0])
    p4 = (RandX + PatchSize[1], RandY)
    
    AllPts = [p1, p2, p3, p4]
    
    # Predict output with forward pass, MiniBatchSize for Test is 1
    # prHVal = EVHomographyNetUnsup(PatchPH, PatchSize, 1)
    # prHVal = tf.reshape(prHVal, (-1, 8, 1))
    
    # HMatPred = stn.solve_DLT(1, AllPts, prHVal)
    # HMatTrue = stn.solve_DLT(1, AllPts, prHTruePH)
   
    # # Warp I1 to I2
    # out_size = [PatchSize[0], PatchSize[1]]
    # WarpI1 = stn.transform(out_size, HMatPred, 1, I1PH)

    # Predict output with forward pass
    pInit = tf.zeros([MiniBatchSize, 8])
    prHVal, WarpI1Patch = ICSTN(PatchPH, PatchSize, MiniBatchSize, opt, pInit)
    # WarpI1Patch = warp.transformImage(opt, I1PH, prHVal)
    
    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        tu.FindNumParams(1)
        # PredOuts = open(WritePath + os.sep + 'PredOuts.txt', 'w')
        NumFilesInCurrDir = len(glob.glob(ReadPath + os.sep + '*' + ImageFormat))
        # Create Write Folder if doesn't exist
        if(not os.path.exists(WritePath)):
            os.makedirs(WritePath)

        for ImgPath in tqdm(glob.glob(ReadPath + os.sep + '*' + ImageFormat)):
            # for StackNum in range(0, NumImgsStack):
            INow = cv2.imread(ImgPath)
            Rho = [25]# [10, 10]
            IBatch, I1Batch, I2Batch, AllPtsBatch, PerturbPtsBatch, H4PtColBatch, MaskBatch = GenerateBatch(INow, Rho, PatchSize, CropType, Vis=False)
            # TODO: Better way is to feed data into a MiniBatch and Extract it again
            # INow = np.hsplit(INow, 2)[0] # Imgs have a stack of 2 in this case, hence extract one
            # prHTrue = np.float32(np.reshape(H4PtColBatch[0], (-1, 4, 2)))
            FeedDict = {PatchPH: IBatch, I1PH: I1Batch, I2PH: I2Batch}# , prHTruePH: prHTrue}
            # prHPredVal, HMatPredVal, HMatTrueVal = sess.run([prHVal, HMatPred, HMatTrue], FeedDict)
            prHValRet, WarpI1PatchRet = sess.run([prHVal, WarpI1Patch], FeedDict)

            print(prHValRet)
            print(H4PtColBatch[0])
            print(np.shape(WarpI1PatchRet))
            cv2.imshow('a', np.squeeze(WarpI1PatchRet[:,:,:,0:3]))
            cv2.imshow('b', I2Batch[0])
            cv2.imshow('c', I1Batch[0])
            cv2.waitKey(0)
            input('q')
            
            # Pixel Error in H4Pt
            ErrorNow = np.sum(np.sqrt((prHPredVal[:, 0] - prHTrue[:, 0])**2 + (prHPredVal[:, 1] - prHTrue[:, 1])**2))/4
            
            # Predicted Rotation
            HMatPredValRot = HMatPredVal[0].copy()
            HMatPredValRot[:, 2] = np.cross(HMatPredValRot[:,0], HMatPredValRot[:,1]).transpose()
            RotPred = mu.ClosestRotMat(HMatPredValRot)
            
            # GT Rotation
            HMatTrueValRot = HMatTrueVal[0].copy()
            HMatTrueValRot[:, 2] = np.cross(HMatTrueValRot[:,0], HMatTrueValRot[:,1]).transpose()
            RotTrue = mu.ClosestRotMat(HMatTrueValRot)
            # Need to handle case when det becomes -1.0

            # Predicted and GT Translation
            TransPred = HMatPredVal[0].copy()
            TransPred = TransPred[:, 2]
            TransTrue = HMatTrueVal[0].copy()
            TransTrue = TransTrue[:, 2]

            print('H4Pt Avg, Error {}'.format(ErrorNow))
            print('H4Pt Pred \n {}'.format(prHPredVal))
            print('H4Pt True \n {}'.format(prHTrue))
            print("HMatPred \n {}".format(HMatPredVal[0]))
            print("HMatTrue \n {}".format(HMatTrueVal[0]))

            print('RotPred \n {}'.format(RotPred))
            print('RotTrue \n {}'.format(RotTrue))

            print('RotPred Det {} RotTrue Det {}'.format(np.linalg.det(RotPred), np.linalg.det(RotTrue)))
            print('Rotation Error {}'.format(mu.RotMatError(RotPred, RotTrue)))
            
            print('TransPred \n {}'.format(TransPred))
            print('TransTrue \n {}'.format(TransTrue))
            print('Translation Error {}'.format(mu.TransError(TransPred, TransTrue)))
            input('a')
            # Timer1 = tic()
            # WarpI1Ret = sess.run(WarpI1, FeedDict)
            # cv2.imshow('a', WarpI1Ret[0])
            # cv2.imshow('b', I1Batch[0])
            # cv2.imshow('c', I2Batch[0])
            # cv2.imshow('d', np.abs(WarpI1Ret[0]- I2Batch[0]))
            # cv2.waitKey(0)
            
            # print(toc(Timer1))
            
            # WarpI1Ret = WarpI1Ret[0]
            # Remap to [0,255] range
            # WarpI1Ret = np.uint8(remap(WarpI1Ret, 0.0, 255.0, np.amin(WarpI1Ret), np.amax(WarpI1Ret)))
            # Crop out junk pixels as they are appended in top left corner due to padding
            # WarpI1Ret = WarpI1Ret[-PatchSize[0]:, -PatchSize[1]:, :]
            
            # IStacked = np.hstack((WarpI1Ret, I2Batch[0]))
            # Write Image to file
            # cv2.imwrite(CurrWritePath + os.sep + 'events' + os.sep +  "event_%d"%(ImgNum+1) + '.png', IStacked)
            
            # Extract Image Name
            # https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
            # Delimiters = ImageFormat, "_" 
            # RegexPattern = '|'.join(map(re.escape, Delimiters))
            # ImgNameNow = re.split(RegexPattern, ImgPath)
            # ImageNum = int(ImgNameNow[-2])
            
            # INow = cv2.imread(ReadPath + os.sep + Prefix%(ImageNum) + ImageFormat)
            # cv2.imwrite(CurrWritePath + os.sep +  Prefix%(ImageNum) + ImageFormat, INow)
            # PredOuts.write(ImgPath + '\t' + str(ErrorNow) + '\n')
    # PredOuts.close()
                    
                    
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

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    ReadPath = Args.ReadPath
    WritePath = Args.WritePath
    GPUDevice = Args.GPUDevice
    CropType = Args.CropType
    MiniBatchSize = 1
    
    # Set GPUNum
    tu.SetGPU(GPUDevice)

    # Setup all needed parameters including file reading
    TrainNames, ImageSize, PatchSize, NumTrainSamples = SetupAll(ReadPath)
    
    class Options:
        def __init__(self, PatchSize=[128,128,3], MiniBatchSize=MiniBatchSize, warpType='homography', NumBlocks=4, pertScale=0.25, transScale=0.25):
            self.W = PatchSize[0].astype(np.int32) # PatchSize is Width, Height, NumChannels
            self.H = PatchSize[1].astype(np.int32) 
            self.batchSize = np.array(MiniBatchSize).astype(np.int32) 
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
            self.NumBlocks = NumBlocks
            self.pertScale = pertScale
            self.transScale = transScale

    opt = Options(PatchSize=PatchSize)
     
    # Define PlaceHolder variables for Input and Predicted output
    PatchPH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], PatchSize[2]*2), name='Input')
    I1PH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], PatchSize[2]), name='I1')
    I2PH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], PatchSize[2]), name='I2')
    prHTruePH = tf.placeholder(tf.float32, shape=(1, 4, 2), name='prHTrue')

    if(not os.path.exists(WritePath)):
        cprint("WARNING: %s doesnt exist, Creating it."%WritePath, 'yellow')
        os.mkdir(WritePath)

    TestOperation(PatchPH, I1PH, I2PH, prHTruePH, PatchSize, ModelPath, ReadPath, WritePath, TrainNames, NumTrainSamples, CropType, MiniBatchSize, opt)
     
if __name__ == '__main__':
    main()
 
