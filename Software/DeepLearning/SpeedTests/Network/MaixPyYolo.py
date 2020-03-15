import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def MaixPyYolo(Img, ImageSize, MiniBatchSize):
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
    conv1 = tf.layers.conv2d(inputs=Img, filters=24, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn1 = tf.nn.relu(conv1)

    conv2 = tf.layers.conv2d(inputs=bn1, filters=24, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn2 = tf.nn.relu(conv2)

    conv3 = tf.layers.conv2d(inputs=bn2, filters=48, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn3 = tf.nn.relu(conv3)

    conv4 = tf.layers.conv2d(inputs=bn3, filters=48, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn4 = tf.nn.relu(conv4)

    conv5 = tf.layers.conv2d(inputs=bn4, filters=96, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn5 = tf.nn.relu(conv5)

    conv6 = tf.layers.conv2d(inputs=bn5, filters=96, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn6 = tf.nn.relu(conv6)

    conv7 = tf.layers.conv2d(inputs=bn6, filters=96, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn7 = tf.nn.relu(conv7)

    conv8 = tf.layers.conv2d(inputs=bn7, filters=96, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn8 = tf.nn.relu(conv8)

    conv9 = tf.layers.conv2d(inputs=bn8, filters=192, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn9 = tf.nn.relu(conv9)

    conv10 = tf.layers.conv2d(inputs=bn9, filters=192, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn10 = tf.nn.relu(conv10)

    conv11 = tf.layers.conv2d(inputs=bn10, filters=192, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn11 = tf.nn.relu(conv11)

    conv12 = tf.layers.conv2d(inputs=bn11, filters=192, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn12 = tf.nn.relu(conv12)

    conv13 = tf.layers.conv2d(inputs=bn12, filters=384, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn13 = tf.nn.relu(conv13)

    conv14 = tf.layers.conv2d(inputs=bn13, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn14 = tf.nn.relu(conv14)

    conv15 = tf.layers.conv2d(inputs=bn14, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn15 = tf.nn.relu(conv15)

    conv16 = tf.layers.conv2d(inputs=bn15, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn16 = tf.nn.relu(conv16)

    conv17 = tf.layers.conv2d(inputs=bn16, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn17 = tf.nn.relu(conv17)

    conv18 = tf.layers.conv2d(inputs=bn17, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn18 = tf.nn.relu(conv18)

    conv19 = tf.layers.conv2d(inputs=bn18, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn19 = tf.nn.relu(conv19)

    conv20 = tf.layers.conv2d(inputs=bn19, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn20 = tf.nn.relu(conv20)

    conv21 = tf.layers.conv2d(inputs=bn20, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn21 = tf.nn.relu(conv21)

    conv22 = tf.layers.conv2d(inputs=bn21, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn22 = tf.nn.relu(conv22)

    conv23 = tf.layers.conv2d(inputs=bn22, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn23 = tf.nn.relu(conv23)

    conv24 = tf.layers.conv2d(inputs=bn23, filters=384, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn24 = tf.nn.relu(conv24)

    conv25 = tf.layers.conv2d(inputs=bn24, filters=768, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn25 = tf.nn.relu(conv25)

    conv26 = tf.layers.conv2d(inputs=bn25, filters=768, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn26 = tf.nn.relu(conv26)

    conv27 = tf.layers.conv2d(inputs=bn26, filters=30, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn27 = tf.nn.relu(conv27)

    conv28 = tf.layers.conv2d(inputs=bn27, flters=30, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=None)#, kernel_initializer=KernelInit)
    bn28 = tf.nn.relu(conv28)

    return bn28

