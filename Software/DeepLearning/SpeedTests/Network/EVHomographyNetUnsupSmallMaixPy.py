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

    # Input = tf.identity(Img, name='Input')
    # conv1 output is of size M/2 x N/2 x 64
    conv1 = tf.layers.conv2d(inputs=Img, filters=16, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv1T')#, kernel_initializer=KernelInit)

    # bn1 output is of size M/2 x N/2 x 64
    bn1 = tf.nn.relu(conv1, name='relu1T')

    conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv2T')#, kernel_initializer=KernelInit)

    # bn1 output is of size M/2 x N/2 x 64
    bn2 = tf.nn.relu(conv2, name='relu2T')

    conv3 = tf.layers.conv2d(inputs=bn2, filters=16, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv3T')#, kernel_initializer=KernelInit)

    # bn1 output is of size M/2 x N/2 x 64
    bn3 = tf.nn.relu(conv3, name='relu3T')

    conv4 = tf.layers.conv2d(inputs=bn3, filters=16, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv4T')#, kernel_initializer=KernelInit)

    # bn1 output is of size M/2 x N/2 x 64
    bn4 = tf.nn.relu(conv4, name='relu4T')
    
    # flat is of size 1 x M/16*N/16*128
    Shape = bn4.get_shape().as_list()       
    Dim = np.prod(Shape[1:])     
    flat = tf.reshape(bn4, [-1, Dim])

    # print(Dim)
    # input('q')
    # Output = tf.reshape(bn1, [-1, 65536])
        
    # fc1 
    Output = tf.layers.dense(flat, units=1000, activation=None, name='fc1T')

    # fc2
    # Output = tf.contrib.layers.fully_connected(flat, num_outputs=8, activation_fn=None)#, name='fc2T')

    # Output = tf.identity(fc2, name='Output')

    return Output

