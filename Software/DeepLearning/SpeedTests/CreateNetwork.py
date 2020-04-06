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
from Network.EVHomographyNetUnsupSmall import EVHomographyNetUnsupSmall
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
import Misc.MiscUtils as mu
import importlib
import Misc.warpICSTN2 as warp2

# Don't generate pyc codes
sys.dont_write_bytecode = True
    
def GenerateModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, MiniBatchSize, NumImgs, Args, Net, opt):    
    # Predict output with forward pass
    # ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
    # Network = getattr(Net, ClassName)
    # VN = Network(InputPH = ImgPH, Training = True, Opt = opt, InitNeurons = Args.InitNeurons)
    # _, prVal, _ = VN.Network()
    prVal = EVHomographyNetUnsupSmall(ImgPH, ImageSize, MiniBatchSize)
            
    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:       
        sess.run(tf.global_variables_initializer())
        print('New model initialized....')

        # Summary_writer = tf.summary.FileWriter("logs_viz",graph=tf.get_default_graph())

        # Print Number of parameters in the network    
        tu.FindNumParams(1)

        # Print out Number of Flops
        NumFlops = tu.FindNumFlops(sess, 1)

        # Print Model Size in MB
        A = tu.CalculateModelSize(1)

        # TFLite Conversions
        def representative_dataset_gen():
            for _ in range(10):
                # Get sample input data as a numpy array in a method of your choosing.
                input = np.float32(2.*(np.random.rand(MiniBatchSize, ImageSize[0], ImageSize[1], NumImgs*ImageSize[2]) - 0.5))
                yield [input]

        if(Args.TFLite):
            converter = tf.lite.TFLiteConverter.from_session(sess, [ImgPH], [prVal])
            # Optimizer Flags
            if(Args.TFLiteOpt == 'Latency'):
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY] 
            elif(Args.TFLiteOpt == 'Size'):
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            else:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Weight Quantization
            if(Args.TFLiteQuant == 'Float16'): # Float32, Float16, Int64, Int32, Int8, UInt8
                # converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
                converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            elif(Args.TFLiteQuant == 'Int64'):
                # converter.target_spec.supported_types = [tf.lite.constants.INT64]
                converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            elif(Args.TFLiteQuant == 'Int32'):
                # converter.target_spec.supported_types = [tf.lite.constants.INT32]
                converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            elif(Args.TFLiteQuant == 'Int8'):
                # converter.target_spec.supported_types = [tf.lite.constants.INT8]
                converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            elif(Args.TFLiteQuant == 'UInt8'):
                # converter.target_spec.supported_types = [tf.lite.constants.QUANTIZED_UINT8]
                converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            else:
                # converter.target_spec.supported_types = [tf.lite.constants.FLOAT]
                converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

            if('Int' in Args.TFLiteQuant):
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            # else:
            #     converter.inference_input_type = tf.float32
            #     converter.inference_output_type = tf.float32
                
            converter.representative_dataset = representative_dataset_gen
            # input_arrays = converter.get_input_arrays()
            # converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
            # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # converter.inference_input_type = tf.uint8
            # converter.inference_output_type = tf.uint8
            tflite_model = converter.convert()
            TFLiteName = Args.CheckPointPath + os.sep + ModelPrefix +'TFLite.tflite'
            open(TFLiteName, "wb").write(tflite_model)
            print('TFLite Model Written in {}....'.format(TFLiteName))

        if(Args.EdgeTPU):
            converter = tf.lite.TFLiteConverter.from_session(sess, [ImgPH], [prVal])
            # Optimizer Flags
            if(Args.TFLiteOpt == 'Latency'):
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY] 
            elif(Args.TFLiteOpt == 'Size'):
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            else:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Weight Quantization
            converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            converter.representative_dataset = representative_dataset_gen

            tflite_model = converter.convert()
            TFLiteName = Args.CheckPointPath + os.sep + ModelPrefix +'TFLiteEdgeTPU.tflite'
            open(TFLiteName, "wb").write(tflite_model)
            print('TFLite Model Written in {}....'.format(TFLiteName))

        # Save model every epoch
        SaveName = CheckPointPath + os.sep + ModelPrefix +'model.ckpt'
        Saver.save(sess, save_path=SaveName)
        print('Model Saved in {}...'.format(SaveName))

    # Reset graph after using: https://github.com/tensorflow/tensorflow/issues/19731
    tf.reset_default_graph()

def SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, MiniBatchSize, TestMultipleMiniBatch,\
                   NumTest, MiniBatchSizeIncrement, MiniBatchMin, NumImgs, Args, Net, opt, Append=False):
    if(TestMultipleMiniBatch):
        StartIdx = MiniBatchMin - 1
    else:
        StartIdx =  MiniBatchSize - 1 # Run 1 iteration only

    WarmUp = 5

    if(Args.TFLite):
        import tflite_runtime.interpreter as tflite
        TFLiteName = Args.CheckPointPath + os.sep + ModelPrefix +'TFLite.tflite'
        # Load TFLite model and allocate tensors.
        interpreter = tflite.Interpreter(model_path=TFLiteName)
        interpreter.allocate_tensors()
        LogFile = open(CheckPointPath + os.sep + ModelPrefix + 'LogTFLite.txt', 'w')

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        if('Float' in Args.TFLiteQuant):
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        else:
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        for count in range(NumTest+WarmUp):
            if(count == WarmUp):
                # Start timer after first iteration
                Timer1 = mu.tic()
            interpreter.invoke()
            # output_data = interpreter.get_tensor(output_details[0]['index'])
        Time = mu.toc(Timer1)/NumTest
        FPS = 1/Time
        LogFile.write('{}, {}, {}\n'.format(1, Time, FPS))
        LogFile.close()
        
    # Open file for logging
    if(not Append):
        LogFile = open(CheckPointPath + os.sep + ModelPrefix + 'Log.txt', 'w')
    else:
        LogFile = open(CheckPointPath + os.sep + ModelPrefix + 'Log.txt', 'a')

    for count2 in range(StartIdx, MiniBatchSize, MiniBatchSizeIncrement):
        MiniBatchSizeNow = count2 + 1
        print('Testing MiniBatchSize {} ....'.format(MiniBatchSizeNow))
        ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSizeNow, ImageSize[0], ImageSize[1], NumImgs*ImageSize[2]), name='Input')
            
        # Predict output with forward pass
        ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
        Network = getattr(Net, ClassName)
        VN = Network(InputPH = ImgPH, Training = True, Opt = opt, InitNeurons = Args.InitNeurons)
        _, prVal, _ = VN.Network()

        # Setup Saver
        Saver = tf.train.Saver()

        with tf.Session() as sess:       
            sess.run(tf.global_variables_initializer())
            print('Model initialized....')

            # Predict Model Size
            ModelSize = 0#tu.CalculateModelSize(sess, PrintFlag=1)

            # Write this the first time only
            if((count2 == StartIdx) and (not Append)):
                # Print Number of parameters in the network    
                NumParams = tu.FindNumParams(1)
                # Print Number of FLOPs in the network
                NumFLOPs = tu.FindNumFlops(sess, 1)

                # Write values to file
                LogFile.write('NumParams {}, NumFLOPs {}, Model Size (MB) {}\n'.format(NumParams, NumFLOPs, ModelSize))
                LogFile.write('MiniBatchSize, Avg. time (s), Eff. Frame Rate (fps) \n')
            
            # Load model
            LoadName = CheckPointPath + os.sep + ModelPrefix +'model.ckpt'
            Saver.restore(sess, LoadName)
            # Extract only numbers from the name
            print('Loaded weights from ' + LoadName + '....')

            # Run once to setup graph
            RandImg = np.random.rand(ImageSize[0], ImageSize[1], NumImgs*ImageSize[2])
            RandImg = RandImg[np.newaxis, :, :, :]

            RandImgBatch = np.tile(RandImg, (MiniBatchSizeNow,1,1,1))
            for count in range(NumTest+WarmUp):
                if(count == WarmUp):
                    # Start timer after first iteration
                    Timer1 = mu.tic()
                FeedDict = {VN.InputPH: RandImgBatch}
                _ = sess.run([prVal], FeedDict)

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
    Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--NetworkName', default='Network.VanillaNet', help='Name of network file, Default: Network.ResNet')
    Parser.add_argument('--TestMultipleMiniBatch', type=int, default=0, help='Used for Testing only, if active MiniBatchSize represents max value')
    Parser.add_argument('--NumTest', type=int, default=100, help='Number of times code will run to take avg. time')
    Parser.add_argument('--Mode', default='Train', help='Mode: Train or Test or TrainTest')
    Parser.add_argument('--MiniBatchMin', type=int, default=1, help='Used for Testing only, is active when TestMultipleMiniBatch is used and MiniBatchMin represents min value.')
    Parser.add_argument('--MiniBatchSizeIncrement', type=int, default=1, help='Used for Testing only, is active when TestMultipleMiniBatch is used and MiniBatchSize represents max value. The code is tested for these increment steps. ')
    Parser.add_argument('--ForceBatchSize1', type=int, default=0, help='Used for Testing only, is active when TestMultipleMiniBatch is used and MiniBatchSize represents max value. Force runs for BatchSize of 1.')
    Parser.add_argument('--TFLite', default=0, type=int, help='Convert to TFLite quantized to type --TFLiteQuant? Default: 0')
    Parser.add_argument('--TFLiteOpt', default='Default', help='TFLite Optimization, Choose from Default, Size and Latency. Default: Default')
    Parser.add_argument('--TFLiteQuant', default='Float32', help='TFLite Quantization. Choose from Float32, Float16, Int64, Int32, Int8, UInt8. Default: Float32')
    Parser.add_argument('--EdgeTPU', default=0, type=int, help='TFLite For EdgeTPU. This works in addition to TFLite conversion. Default: 0')
    Parser.add_argument('--InitNeurons', type=float, default=8, help='Learning Rate, Default: 8')

    Args = Parser.parse_args()
    CheckPointPath = Args.CheckPointPath
    ModelPrefix = Args.ModelPrefix
    GPUDevice = Args.GPUDevice
    MiniBatchSize = Args.MiniBatchSize
    Mode = Args.Mode
    TestMultipleMiniBatch = bool(Args.TestMultipleMiniBatch)
    NumTest = Args.NumTest
    MiniBatchSizeIncrement = Args.MiniBatchSizeIncrement
    MiniBatchMin = Args.MiniBatchMin
    ForceBatchSize1 = bool(Args.ForceBatchSize1)

    # Import Network Module
    Net = importlib.import_module(Args.NetworkName)

    # Parameters
    ImageSize = np.array([128, 128, 3])
    NumImgs = 1

    warpType = ['pseudosimilarity']
    if(len(warpType) > 1):
        print('ERROR: Only 1 warping block is supported in TFLite conversions.')
        sys.exit()
    opt = warp2.Options(PatchSize=ImageSize, MiniBatchSize=MiniBatchSize, warpType = warpType) # ICSTN Options

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)
    
    # Set GPUDevice
    tu.SetGPU(GPUDevice)

    # Placeholders
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], NumImgs*ImageSize[2]), name='Input')
    
    # Generate the model and save        
    if(Mode == 'Train'):
        GenerateModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, MiniBatchSize, NumImgs, Args, Net, opt)
    elif(Mode == 'Test'):
        if(ForceBatchSize1 and TestMultipleMiniBatch):
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, 1, False,\
                           NumTest, MiniBatchSizeIncrement, 1, NumImgs, Args, Net)
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, MiniBatchSize,\
                           TestMultipleMiniBatch, NumTest, MiniBatchSizeIncrement, MiniBatchMin, NumImgs, Args, Net, opt, Append=True)
        else:
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, MiniBatchSize,\
                           TestMultipleMiniBatch, NumTest, MiniBatchSizeIncrement, MiniBatchMin, NumImgs, Args, Net, opt)
    elif(Mode=='TrainTest'):
        GenerateModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, MiniBatchSize, NumImgs, Args, Net, opt)
        if(ForceBatchSize1 and TestMultipleMiniBatch):
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, 1, False\
                           , NumTest, MiniBatchSizeIncrement, 1, NumImgs, Args, Net, opt)
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, MiniBatchSize,\
                           TestMultipleMiniBatch, NumTest, MiniBatchSizeIncrement, MiniBatchMin, NumImgs, Args, Net, opt, Append=True)
        else:
            SpeedTestModel(ImgPH, ImageSize, CheckPointPath, ModelPrefix, MiniBatchSize,\
                           TestMultipleMiniBatch, NumTest, MiniBatchSizeIncrement, MiniBatchMin, NumImgs, Args, Net, opt)
    else:
        print('ERROR: Invalid Mode!')
        sys.exit(0)
        
    
if __name__ == '__main__':
    main()
