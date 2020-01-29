import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps

# Don't generate pyc codes
sys.dont_write_bytecode = True

# TODO: Auto append names
# TODO: Decorator to get function name
# TODO: Add training flag

    
class MakeNet(object):
    def __init__(self):
        self.CurrBlock = 0
    # Decorator to count number of functions have been called
    # Ideas from
    # https://stackoverflow.com/questions/13852138/how-can-i-define-decorator-method-inside-class
    # https://stackoverflow.com/questions/41678265/how-to-increase-a-number-every-time-a-function-is-run
    def Count(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            self.CurrBlock += 1
            return func(self, *args, **kwargs)
        return wrapped

    def CountAndScope(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            with tf.variable_scope(func.__name__ + str(self.CurrBlock)):
                self.CurrBlock += 1
                return func(self, *args, **kwargs)
        return wrapped
    
    @CountAndScope
    def ConvBNReLUBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        conv =  self.Conv(inputs, filters, kernel_size, strides, padding)
        bn = self.BN(conv)
        Output = self.ReLU(bn)
        return Output
        
    @CountAndScope
    def Conv(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        Output = tf.layers.conv2d(inputs = inputs, filters = filters, kernel_size = kernel_size,\
                                  strides = strides, padding = padding, activation=None) 
        return Output

    @CountAndScope
    def BN(self, inputs = None):
        Output = tf.layers.batch_normalization(inputs = inputs) 
        return Output

    @CountAndScope
    def ReLU(self, inputs = None):
        Output = tf.nn.relu(inputs)
        return Output

class SqueezeNet(MakeNet):
    def __init__(self, ImageSize = None, NumOut = None):
        super(SqueezeNet, self).__init__()
        self.ImageSize = ImageSize
        self.NumOut = NumOut

    def MakeNet(self):
        ImgPH = tf.placeholder(tf.float32, shape=(32, 100, 100, 3), name='Input')
        for count in range(4):
            if(count == 0):
                Net = ImgPH
            else:
                Net = self.ConvBNReLUBlock(inputs = Net, filters = 6, kernel_size = (3,3), strides = (2,2), padding = 'same')
        
SN = SqueezeNet()
print(SN.CurrBlock)
Z = SN.MakeNet()
print(SN.CurrBlock)
Writer = tf.summary.FileWriter('/home/nitin/PRGEye/Logs3/', graph=tf.get_default_graph())        


# def main():
#     SN = SqueezeNet()


# if __name__=="__main__":
#     main()
