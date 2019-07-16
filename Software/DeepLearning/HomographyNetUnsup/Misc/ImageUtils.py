import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import tensorflow as tf
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CenterCrop(I, OutShape):
    ImageSize = np.shape(I)
    CenterX = ImageSize[0]/2
    CenterY = ImageSize[1]/2
    try:
        ICrop = I[int(np.ceil(CenterX-OutShape[0]/2)):int(np.ceil(CenterX+OutShape[0]/2)),\
                  int(np.ceil(CenterY-OutShape[1]/2)):int(np.ceil(CenterY+OutShape[1]/2)), :]
    except:
        ICrop = None
    return ICrop

def CenterCropFactor(I, Factor):
    ImageSize = np.shape(I)
    CenterX = ImageSize[0]/2
    CenterY = ImageSize[1]/2
    OutShape = ImageSize - (np.mod(ImageSize,2**Factor))
    OutShape[2] = ImageSize[2]
    try:
        ICrop = I[int(np.ceil(CenterX-OutShape[0]/2)):int(np.ceil(CenterX+OutShape[0]/2)),\
                  int(np.ceil(CenterY-OutShape[1]/2)):int(np.ceil(CenterY+OutShape[1]/2)), :]
    except:
        ICrop = None
        OutShape = None
    return (ICrop, OutShape)

def RandomCrop(I1, OutShape):
    ImageSize = np.shape(I1)
    try:
        RandX = random.randint(0, ImageSize[0]-OutShape[0])
        RandY = random.randint(0, ImageSize[1]-OutShape[1])
        I1Crop = I1[RandX:RandX+OutShape[0], RandY:RandY+OutShape[1], :]
    except:
        I1Crop = None
    return (I1Crop)

def GaussianNoise(I1):
    IN1 = skimage.util.random_noise(I1, mode='gaussian', var=0.01)
    IN1 = np.uint8(IN1*255)
    return (IN1)

def ShiftHue(I1):
    IHSV1 = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)
    MaxShift = 30
    RandShift = random.randint(-MaxShift, MaxShift)
    IHSV1[:, :, 0] = IHSV1[:, :, 0] + RandShift
    IHSV1 = np.uint8(np.clip(IHSV1, 0, 255))
    return (cv2.cvtColor(IHSV1, cv2.COLOR_HSV2BGR))

def ShiftSat(I1):
    IHSV1 = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)
    MaxShift = 30
    RandShift = random.randint(-MaxShift, MaxShift)
    IHSV1 = np.int_(IHSV1)
    IHSV1[:, :, 1] = IHSV1[:, :, 1] + RandShift
    IHSV1 = np.uint8(np.clip(IHSV1, 0, 255))
    return (cv2.cvtColor(IHSV1, cv2.COLOR_HSV2BGR))

def Gamma(I1):
    MaxShift = 2.5
    RandShift = random.uniform(0, MaxShift)
    IG1 = skimage.exposure.adjust_gamma(I1, RandShift)
    return (IG1)

def Resize(I1, OutShape):
    ImageSize = np.shape(I1)
    I1Resize = cv2.resize(I1, (OutShape[0], OutShape[1]))
    return (I1Resize)

def StandardizeInputs(I):
    I /= 255.0
    I -= 0.5
    I *= 2.0
    return I

def StandardizeInputsTF(I):
    I = tf.math.multiply(tf.math.subtract(tf.math.divide(I, 255.0), 0.5), 2.0)
    return I

def HPFilter(I, Radius = 10):
    # Code adapted from: https://akshaysin.github.io/fourier_transform.html#.XSYBbnVKhhF
    if(len(np.shape(I)) == 3):
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    F = cv2.dft(np.float32(I), flags=cv2.DFT_COMPLEX_OUTPUT)
    FShift = np.fft.fftshift(F)
       
    # Circular HPF mask, center circle is 0, remaining all ones
    Rows, Cols = I.shape
    Mask = np.ones((Rows, Cols, 2), np.uint8)
    Center = [int(Rows / 2), int(Cols / 2)]
    x, y = np.ogrid[:Rows, :Cols]
    MaskArea = (x - Center[0]) ** 2 + (y - Center[1]) ** 2 <= Radius**2
    Mask[MaskArea] = 0

    # Filter by Masking FFT Spectrum
    FShiftFilt = np.multiply(FShift, Mask)
    FFilt = np.fft.ifftshift(FShiftFilt)
    IFilt = cv2.idft(FFilt)
    IFilt = cv2.magnitude(IFilt[:, :, 0], IFilt[:, :, 1])
    return IFilt

def rgb2gray(rgb):
    # Code adapted from: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
