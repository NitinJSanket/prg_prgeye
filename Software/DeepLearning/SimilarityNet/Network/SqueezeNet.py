#!/usr/bin/env python

# Run as python -m SimilarityNet.Network.SqueezeNet from DeepLearning folder

import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
# Required to import ..Misc so you don't have to run as package with -m flag
# sys.path.insert(0, '../Misc/')
# import TFUtils as tu
# from Decorators import *
from ..Misc import TFUtils as tu
from ..Misc.Decorators import *
import ..Misc.warpICSTN2 as warp2
from BaseLayers import *

# TODO: Add training flag

class SqueezeNet(BaseLayers):
    def __init__(self, InputPH = None, Training = False,  Padding = None, Opt = None, NumFire = None, NumFireConvModules = None):
        super(SqueezeNet, self).__init__()
        if(InputPH is None):
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        if( Opt is None):
            print('ERROR: Options cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        self.InitNeurons = 8
        self.Training = Training
        self.ExpansionFactor = 2.0
        self.DropOutRate = 0.7
        if(padding is None):
            padding = 'same'
        self.Padding = Padding
        self.Opt = Opt
        if(NumFire is None):
            NumFire =  2
        if(NumFireConv is None):
            NumFireConv = 3
        self.NumFireConv = NumFireConv
        self.NumFire = NumFire

    @CountAndScope
    @add_arg_scope
    def OutputLayer(self, inputs = None, padding = None, rate=None, NumOut=None):
        if(rate is None):
            rate = 0.5
        if(NumOut is None):
           NumOut = self.NumOut     
        flat = self.Flatten(inputs = inputs)
        drop = self.Dropout(inputs = flat, rate=rate)
        dense = self.Dense(inputs = drop, filters = NumOut, activation=None)
        return dense

    @CountAndScope
    @add_arg_scope
    def FireModule(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, Bypass = False):
        expandfilter = int(4.0*filters)
        squeeze = self.Conv(inputs = inputs, filters = filters, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu, name='squeeze')
        expand1x1 = self.Conv(inputs = squeeze, filters = expandfilter, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu, name='expand1x1')
        expand3x3 = self.Conv(inputs = squeeze, filters = expandfilter, kernel_size = (3,3), padding = padding, strides=(1,1), activation=tf.nn.relu, name='expand3x3')
        concat = self.Concat(inputs = [expand1x1, expand3x3])
        if(Bypass):
            concat = tf.math.add(inputs, concat, name='add')
        return concat

    @CountAndScope
    @add_arg_scope
    def FireConvBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, Bypass = False, NumFire = None):
        for count in range(NumFire): 
            Net = FireModule(self, inputs = inputs, filters = None, kernel_size = None, strides = None, padding = None, Bypass = False)
        Net = self.Conv(inputs = Net, filters = filters, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu)
        return Net

    @CountAndScope
    @add_arg_scope
    def ICSTNBlock(self, inputs = None, filters = None, NumOut = None):
        # Conv
        Net = self.ConvBNReLUBlock(inputs = inputs, padding = self.Padding, filters = filters, kernel_size = (7,7))
        
        # Conv
        NumFilters = int(filters*self.ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs = Net, padding = self.Padding, filters = NumFilters, kernel_size = (5,5))

        # 3 x FireConv blocks
        for count in range(self.NumFireConv):
            NumFilters = int(NumFilters*self.ExpansionFactor)
            Net = FireConvBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, Bypass = False, NumFire = self.NumFire)

       # TODO: Global Avg. Pool
        # Output
        Net = self.OutputLayer(self, inputs = Net, padding = self.Padding, rate=self.DropOutRate, NumOut = NumOut)
        return Net
        
    def _arg_scope(self):
        with arg_scope([self.Conv,  self.FireConvBlock], kernel_size = (3,3), strides = (2,2), padding = self.Padding) as sc: 
            return sc
        
    def Network(self):
        with arg_scope(self._arg_scope()):
             for count in range(self.Opt.NumBlocks):
                 if(count == 0):
                     pNow = self.Opt.pInit
                     pMtrxNow = warp2.vec2mtrx(self.Opt, pNow)
                # Warp Original Image based on previous composite warp parameters
                ImgWarpNow = warp2.transformImage(self.Opt, self.InputPH, pMtrxNow)

                # Compute current warp parameters
                dpNow = self.ICSTNBlock(self.InputPH,  filters = self.InitNeurons, NumOut = self.Opt.warpDim[count]) 
                dpMtrxNow = warp2.vec2mtrx(self.Opt, dpNow)
                pMtrxNow = warp2.compose(self.Opt, pMtrxNow, dpMtrxNow)

                # Update counter used for looping over warpType
                self.Opt.currBlock += 1
                
            # Decrement counter so you use last warp Type
            self.Opt.currBlock -= 1
            ImgWarp = warp2.transformImage(self.Opt, self.InputPH, pMtrxNow) # Final Image Warp
            pNow = warp2.mtrx2vec(self.opt, pMtrxNow)
            
        return pMtrxNow, pNow, ImgWarp

def main():
   tu.SetGPU(1)
   # Test functionality of code
   PatchSize = np.array([100, 100, 3])
   MiniBatchSize = 32
   InputPH = tf.placeholder(tf.float32, shape=(32, 100, 100, 3), name='Input')
   # Create network class variable
   Opt =  warp2.Options(PatchSize= PatchSize, MiniBatchSize=MiniBatchSize, warpType = ['pseudosimilarity', 'pseudosimilarity']) # ICSTN Options
   SN = SqueezeNet(InputPH = InputPH,  Opt = Opt, NumFire = 2, NumFireConvModules = 3)
   # Build the atual network
   pMtrxNow, pNow, ImgWarp = SN.Network()
   # Setup Saver
   Saver = tf.train.Saver()
   with tf.Session() as sess:
       # Initialize Weights
       sess.run(tf.global_variables_initializer())
       tu.FindNumParams(1)
       tu.CalculateModelSize(1)
       tu.FindNumFlops(sess, 1)
       # Save model every epoch
       SaveName = '/home/nitin/PRGEye/CheckPoints/SpeedTests/TestVanillaNet/model.ckpt'
       Saver.save(sess, save_path=SaveName)
       print(SaveName + ' Model Saved...') 
       input('q')
       FeedDict = {SN.InputPH: np.random.rand(32,100,100,3)}
       RetVal = sess.run([Network], feed_dict=FeedDict) 
   Writer = tf.summary.FileWriter('/home/nitin/PRGEye/Logs3/', graph=tf.get_default_graph())

if __name__=="__main__":
    main()
