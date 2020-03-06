#!/usr/bin/env python

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
import Misc.TFUtils as tu2
import Misc.warpICSTN as warp
import Misc.warpICSTN2 as warp2
from Misc.DataHandling import *
from Misc.BatchCreationNP import *
from Misc.BatchCreationTF import *
from Network.VanillaNet import *
from Misc.Decorators import *

def main():
   tu.SetGPU(-1)
   # Test functionality of code
   PatchSize = np.array([128, 128, 3])
   MiniBatchSize = 1
   InputPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]*2), name='Input')
   # Create network class variable
   Opt =  warp2.Options(PatchSize= PatchSize, MiniBatchSize=MiniBatchSize, warpType = ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity']) # ICSTN Options
   VN = VanillaNet(InputPH = InputPH, Training = True, Opt = Opt)
   # Build the atual network
   pMtrxNow, pNow, ImgWarp = VN.Network()
   # Setup Saver
   Saver = tf.train.Saver()
   # This runs on 1 thread of CPU when tu.SetGPU(-1) is set
   # config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True)
   # tf.Session(config=config)
   with tf.Session() as sess:
       # Initialize Weights
       sess.run(tf.global_variables_initializer())
       tu.FindNumFlops(sess, 1)
       tu.FindNumParams(1)
       tu.CalculateModelSize(1)
       # Save model every epoch
       SaveName = '/home/nitin/PRGEye/CheckPoints/SpeedTests/TestVanillaNet/model.ckpt'
       Saver.save(sess, save_path=SaveName)
       print(SaveName + ' Model Saved...') 
       FeedDict = {VN.InputPH: np.random.rand(MiniBatchSize,PatchSize[0],PatchSize[1],PatchSize[2]*2)}
       for count in range(10):
           Timer1 = mu.tic()
           pMtrxNowVal, pNowVal, ImgWarpVal = sess.run([pMtrxNow, pNow, ImgWarp], feed_dict=FeedDict)
           print(1/mu.toc(Timer1))

       for _ in tqdm(range(100)):
           pMtrxNowVal, pNowVal, ImgWarpVal = sess.run([pMtrxNow, pNow, ImgWarp], feed_dict=FeedDict)

   Writer = tf.summary.FileWriter('/home/nitin/PRGEye/Logs3/', graph=tf.get_default_graph())

if __name__=="__main__":
    main()
