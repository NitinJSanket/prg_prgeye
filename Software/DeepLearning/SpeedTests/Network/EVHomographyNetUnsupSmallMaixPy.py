import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def EVHomographyNetUnsupSmallMaixPy(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3
    # KernelInit = tf.initializers.random_normal(mean=0.0, stddev=1e-3)

    Input = tf.identity(Img, name='Input')
    # conv1 output is of size M/2 x N/2 x 64
    conv1 = tf.layers.conv2d(inputs=Input, filters=16, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv1T')#, kernel_initializer=KernelInit)

    # bn1 output is of size M/2 x N/2 x 64
    bn1 = tf.nn.relu(conv1, name='relu1T')
    
    # flat is of size 1 x M/16*N/16*128
    Shape = bn1.get_shape().as_list()       
    Dim = np.prod(Shape[1:])        
    flat = tf.reshape(bn1, [-1, Dim])
        
    # fc1 
    # fc1 = tf.layers.dense(flat, units=256, activation=None, name='fc1T')

    # fc2
    fc2 = tf.layers.dense(flat, units=8, activation=None, name='fc2T')

    Output = tf.identity(fc2, name='Output')

    return Output

