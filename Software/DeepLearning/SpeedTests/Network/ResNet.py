#!/usr/bin/env python

import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
# Required to import ..Misc so you don't have to run as package with -m flag
import Misc.TFUtils as tu
from Misc.Decorators import *
import Misc.warpICSTN2 as warp2
from Network.BaseLayers import *
import Misc.MiscUtils as mu

# TODO: Add training flag

class ResNet(BaseLayers):
    # http://torch.ch/blog/2016/02/04/resnets.html
    def __init__(self, InputPH = None, Training = False,  Padding = None, Opt = None, NumBlocks = None, InitNeurons = None, Suffix = None):
        super(ResNet, self).__init__()
        if(InputPH is None):
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        if(Opt is None):
            print('ERROR: Options cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        if(InitNeurons is None):
            InitNeurons = 16
        self.InitNeurons = InitNeurons
        self.Training = Training
        self.ExpansionFactor = 2
        self.DropOutRate = 0.7
        if(Padding is None):
            Padding = 'same'
        self.Padding = Padding
        self.Opt = Opt
        if(NumBlocks is None):
            NumBlocks =  3
        self.NumBlocks = NumBlocks
        if(Suffix is None):
            Suffix = ''
        self.Suffix = Suffix

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
    def ResBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        Net = self.ConvBNReLUBlock(inputs = inputs, filters = filters, padding = padding, strides=(1,1))
        Net = self.Conv(inputs = Net, filters = filters, padding = padding, strides=(1,1), activation=None)
        Net = self.BN(inputs = Net)
        Net = tf.add(Net, inputs)
        Net = self.ReLU(inputs = Net)
        return Net

    @CountAndScope
    @add_arg_scope
    def ResNetBlock(self, inputs = None, filters = None, NumOut = None):
        # Conv
        Net = self.ConvBNReLUBlock(inputs = inputs, padding = self.Padding, filters = filters, kernel_size = (7,7))
        
        # Conv
        NumFilters = int(filters*self.ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs = Net, padding = self.Padding, filters = NumFilters, kernel_size = (5,5))

        # 3 x Res blocks
        for count in range(self.NumBlocks):
            Net = self.ResBlock(inputs = Net, filters = NumFilters)
            NumFilters = int(NumFilters*self.ExpansionFactor)
            # Extra Conv for downscaling
            Net = self.Conv(inputs = Net, filters = NumFilters, padding = self.Padding, activation=None)
        
        # Output
        Net = self.OutputLayer(inputs = Net, padding = self.Padding, rate=self.DropOutRate, NumOut = NumOut)
        return Net
        
    def _arg_scope(self):
        with arg_scope([self.Conv, self.ConvBNReLUBlock, self.ResBlock], kernel_size = (3,3), strides = (2,2), padding = self.Padding) as sc: 
            return sc
        
    def Network(self):
        with arg_scope(self._arg_scope()):
            for count in range(self.Opt.NumBlocks):
                if(count == 0):
                    pNow = self.Opt.pInit
                    pMtrxNow = warp2.vec2mtrx(self.Opt, pNow)
                with tf.variable_scope('ICTSNBlock' + str(count)):
                    # Warp Original Image based on previous composite warp parameters
                    if(self.Training):
                        ImgWarpNow = warp2.transformImage(self.Opt, self.InputPH, pMtrxNow)

                    # Compute current warp parameters
                    dpNow = self.ResNetBlock(self.InputPH,  filters = self.InitNeurons, NumOut = self.Opt.warpDim[count])
                    pMtrxNow = None

                    # Update counter used for looping over warpType
                    self.Opt.currBlock += 1

                    if(self.Opt.currBlock == self.Opt.NumBlocks):
                        # Decrement counter so you use last warp Type
                        self.Opt.currBlock -= 1
                        pNow = dpNow
                        ImgWarp = None  
                        
            
        return pMtrxNow, pNow, ImgWarp
