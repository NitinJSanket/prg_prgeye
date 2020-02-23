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
import sys
import os
import glob
import Misc.ImageUtils as iu
import Misc.MiscUtils as mu
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

def ConvertOperation(PatchPH, PerturbParamsPH, PatchSize, ModelPath, WritePath,  warpType, opt, Net, InitNeurons, WriteName):
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
    # Predict output with forward pass
    # Create Network Object with required parameters
    VN = Net.MobileNetv1(InputPH = PatchPH, Training = False, Opt = opt, InitNeurons = InitNeurons)
    # Predict output with forward pass
    prParams = VN.Network()

    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # Restore Model
        # Saver.restore(sess, ModelPath)
        sess.run(tf.global_variables_initializer())
        # Print out Number of parameters
        NumParams = tu.FindNumParams(1)
        # Print out Number of Flops
        NumFlops = tu.FindNumFlops(sess, 1)
        # Print out Expected Model Size
        ModelSize = tu.CalculateModelSize()*3 # For some wierd reason result has to be multiplied by 3
        print('Expected Model Size is %f' % ModelSize)

        def representative_dataset_gen():
            for _ in range(100):
                # Get sample input data as a numpy array in a method of your choosing.
                input = np.float32(2.*(np.random.rand(1, PatchSize[0], PatchSize[1], 2*PatchSize[2]) - 0.5))
                yield [input]

        converter = tf.lite.TFLiteConverter.from_session(sess, [PatchPH], [prParams])
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY] #[tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
        # converter.representative_dataset = representative_dataset_gen
        # input_arrays = converter.get_input_arrays()
        # converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
        # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        FileName = WritePath + os.sep + WriteName + '.tflite'
        open(FileName, "wb").write(tflite_model)
        print('TFLite Model Written in {}....'.format(FileName))

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
    Parser.add_argument('--WritePath', dest='WritePath', default='./',\
                                                                             help='Path to load images from, Default:./')
    Parser.add_argument('--WriteName', dest='WriteName', default='NETWORKNAME',\
                                                                             help='Path to load images from, Default:NETWORKNAME')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--NetworkName', default='Network.MobileNetv1TFLite', help='Name of network file, Default: Network.VanillaNet')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    WritePath = Args.WritePath
    GPUDevice = Args.GPUDevice
    NetworkName = Args.NetworkName
    WriteName = Args.WriteName
    if(WriteName == 'NETWORKNAME'):
        WriteName = NetworkName.split('.')[0]

    # Import Network Module
    Net = importlib.import_module(NetworkName)
    
    # Set GPUNum
    tu.SetGPU(GPUDevice)

    # Setup all needed parameters including file reading
    InitNeurons = 8
    warpType = ['pseudosimilarity', 'pseudosimilarity']#['translation', 'translation', 'scale', 'scale']  # ['pseudosimilarity']
    # Homography Perturbation Parameters
    PatchSize = np.array([128, 128, 3])

    opt = warp2.Options(PatchSize=PatchSize, MiniBatchSize=1, warpType = warpType) # ICSTN Options
     
    # Define PlaceHolder variables for Input and Predicted output
    PatchPH = tf.placeholder(tf.float32, shape=(1, PatchSize[0], PatchSize[1], PatchSize[2]*2), name='Input')
    PerturbParamsPH = tf.placeholder(tf.float32, shape=(1, 3), name='PerturbParams')

    if(not os.path.exists(WritePath)):
        cprint("WARNING: %s doesnt exist, Creating it."%WritePath, 'yellow')
        os.mkdir(WritePath)

    ConvertOperation(PatchPH, PerturbParamsPH, PatchSize, ModelPath, WritePath, warpType, opt, Net, InitNeurons, WriteName)

     
if __name__ == '__main__':
    main()
 
