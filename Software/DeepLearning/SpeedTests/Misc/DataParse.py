#!/usr/bin/env python


# Dependencies:
# opencv, do (pip install opencv-python)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)

import glob
import os
from termcolor import colored, cprint
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import random
import re
import ImageUtils as iu


def CenterCropFactorAllImgs(ReadPath, WritePath, Prefix, ImageFormat, Factor):
    if(not os.path.exists(WritePath)):
            os.makedirs(WritePath)
    for dirs in tqdm(glob.glob(ReadPath + os.sep + '*' + ImageFormat)):
        CurrReadPath = dirs
        # Extract Image Name
        # https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
        Delimiters = ImageFormat, "_"
        RegexPattern = '|'.join(map(re.escape, Delimiters))
        ImgNameNow = re.split(RegexPattern, dirs)
        ImageNum = int(ImgNameNow[-2])
        INow, _ = iu.CenterCropFactor(cv2.imread(ReadPath + os.sep + Prefix%(ImageNum) + ImageFormat), Factor)
        cv2.imwrite(WritePath + os.sep +  Prefix%(ImageNum) + ImageFormat, INow)

    
def MakeImgPairs(ReadPath, WritePath, Prefix, ImageFormat):
    if(not os.path.exists(WritePath)):
            os.makedirs(WritePath)
            
    DirNames = open(WritePath + os.sep + 'DirNames.txt', 'w')
    for dirs in tqdm(glob.glob(ReadPath + os.sep + '*' + ImageFormat)):
        CurrReadPath = dirs
        # Extract Image Name
        # https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
        Delimiters = ImageFormat, "_"
        RegexPattern = '|'.join(map(re.escape, Delimiters))
        ImgNameNow = re.split(RegexPattern, dirs)
        ImageNum = int(ImgNameNow[-2])
        # INow = cv2.imread(ReadPath + os.sep + Prefix%(ImageNum) + ImageFormat)
        # cv2.imwrite(CurrWritePath + os.sep +  Prefix%(ImageNum) + ImageFormat, INow)
        DirNames.write(Prefix%(ImageNum) + ImageFormat + '\n')
    DirNames.close()

def SetupSplits(Ratios, WritePath): 
    """
    Inputs: 
    Ratios: 3x1 list of ratio for Train, Validation and Test, each value lies between [0,1]
    Outputs:
    Writes 3 text files ./TxtFiles/Train.txt, ./TxtFiles/Val.txt and ./TxtFiles/Test.txt with Idxs corresponding to ./TxtFiles/DirNames.txt
    """
    # Ratios is a list of ratios from [0,1] for Training, Validation and Testing
    DirFiles = ReadDirNames(WritePath + os.sep + 'DirNames.txt')
    NumFiles = len(DirFiles)
    RandIdxs = range(NumFiles)
    random.shuffle(RandIdxs)
    TrainIdxs = RandIdxs[0:int(np.floor(NumFiles*Ratios[0]))]
    ValIdxs = RandIdxs[int(np.floor(NumFiles*Ratios[0])+1):int(np.floor(NumFiles*Ratios[0]+NumFiles*Ratios[1]))]
    TestIdxs = RandIdxs[int(np.floor(NumFiles*Ratios[0]+NumFiles*Ratios[1])):int(NumFiles-1)]

    if(not (os.path.isfile(WritePath + os.sep + 'Train.txt'))):
        Train = open(WritePath + os.sep + 'Train.txt', 'w')
        for TrainNum in range(len(TrainIdxs)):
            Train.write(str(TrainIdxs[TrainNum])+'\n')
        Train.close()
    else:
        cprint('WARNING: Train.txt File exists', 'yellow')

    if(not (os.path.isfile(WritePath + os.sep + 'Val.txt'))):
        Val = open(WritePath + os.sep + 'Val.txt', 'w')
        for ValNum in range(len(ValIdxs)):
            Val.write(str(ValIdxs[ValNum])+'\n')
        Val.close()
    else:
        cprint('WARNING: Val.txt File exists', 'yellow')

    if(not (os.path.isfile(WritePath + os.sep + 'Test.txt'))):
        Test = open(WritePath + os.sep + 'Test.txt', 'w')
        for TestNum in range(0, len(TestIdxs)):
            Test.write(str(TestIdxs[TestNum])+'\n')
        Test.close()
    else:
        cprint('WARNING: Test.txt File exists', 'yellow')
    # Read Splits once processed
    Train, Val, Test = ReadSplits(WritePath)

    return Train, Val, Test

def ReadDirNames(DirNamesPath):
    """
    Inputs: 
    None
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(DirNamesPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames

def ReadSplits(WritePath):
    """
    Inputs: 
    None
    Outputs:
    Train, Val and Test are data loaded from ./TxtFiles/Train.txt, ./TxtFiles/Val.txt and ./TxtFiles/Test.txt respectively
    They contain the Idxs of all the images to be used for training/validation (held-out testing in this case)/testing
    """
    Train = open(WritePath + os.sep + 'Train.txt', 'r')
    Train = Train.read()
    Train = map(int, Train.split())

    Val = open(WritePath + os.sep + 'Val.txt', 'r')
    Val = Val.read()
    Val = map(int, Val.split())

    Test = open(WritePath + os.sep + 'Test.txt', 'r')
    Test = Test.read()
    Test = map(int, Test.split())

    return Train, Val, Test

def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ReadPath', default='/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/dataset',
                        help='Base path of images, Default:/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/dataset')
    Parser.add_argument('--WritePath', default='/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/processed',
                        help='Base path of images, Default:/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/processed')
    Parser.add_argument('--Prefix', default='COCO_train2014_%012d', help='Prefix name in images, Default: COCO_train2014_%012d')
    Parser.add_argument('--ImageFormat', default='.png', help='Image Format, Default: .png')
    
    Args = Parser.parse_args()
    ReadPath = Args.ReadPath
    WritePath = Args.WritePath
    Prefix = Args.Prefix
    ImageFormat = Args.ImageFormat
    
    if(not os.path.exists(WritePath)):
        cprint("WARNING: %s doesnt exist, Creating it."%WritePath, 'yellow')
        os.mkdir(WritePath)
        
    CenterCropFactorAllImgs(ReadPath, WritePath, Prefix, ImageFormat, Factor=3)
    # MakeImgPairs(ReadPath, WritePath, Prefix, ImageFormat)
    # Ratios = [0.9, 0.00, 0.10]
    # Train, Val, Test = SetupSplits(Ratios, WritePath)
    
if __name__ == '__main__':
    main()
