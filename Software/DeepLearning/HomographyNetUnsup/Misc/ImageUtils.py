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

def StackImages(I1, I2):
    return np.dstack((I1, I2))

def UnstackImages(I, NumChannels=3):
    return I[:,:,:NumChannels], I[:,:,NumChannels:]
    

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

class Homography:
    def __init__(self, ImageSize=[128., 128., 3.]):
        self.ImageSize = ImageSize
        self.ScaleMtrx = np.eye(3) # Scales from [-1, 1] ImageCoordinates to Actual Image Coordinates
        self.ScaleMtrx[0,0] = ImageSize[0]/2
        self.ScaleMtrx[0,2] = ImageSize[0]/2
        self.ScaleMtrx[1,1] = ImageSize[1]/2
        self.ScaleMtrx[1,2] = ImageSize[1]/2
        
    def ComposeHFromRTN(self, R=np.eye(3), T=np.zeros((3, 1)), N= np.array([[0.], [0.], [1.]]), Scale=False):
        H = np.add(R, np.matmul(T, N.T)) # R + TN'
        H = np.divide(H, H[2,2]) # Nornalize by making last element 1
        if(Scale):
            H =  np.matmul(self.ScaleMtrx, np.matmul(H, np.linalg.inv( self.ScaleMtrx))) # Scale to bring to Image Coordinates
           
        return H

    def DecomposeHToRTN(self):
        # retval, rotations, translations, normals   =  cv.decomposeHomographyMat(H, K[, rotations[, translations[, normals]]])
        pass
    
    def WarpImg(self, I, H, Disp=False, DispName='WarpedImg', WaitTime=0):
        WarpedImg = cv2.warpPerspective(I, HInv, (ImageSize[1],ImageSize[0]))
        if(Disp):
            cv2.imshow(DispName, WarpedImg)
            cv2.waitKey(WaitTime)
        return WarpedImg
    
    def WarpPtsUsingHomography(Pts, H, AddOffset=None):
        PerturbPts = []
        for pt in Pts:
            # Apply Homography
            PerturbPtsNow = np.matmul(H, [[pt[0]], [pt[1]], [1.0]])
            # Normalize to be on Image Plane
            PerturbPtsNow = np.divide(PerturbPtsNow, PerturbPtsNow[2])[:2]
            # Add offset if needed
            if(AddOffset is not None):
                PerturbPtsNow = np.add(PerturbPtsNow,  AddOffset)
                PerturbPts.append(PerturbPtsNow)
        return PerturbPts

    def DispWarpLines(self, I, Pts, Disp=True,  DispName='HomographyLines', ColorSpec=(255,255,255), WaitTime=0):
        ImgDisp = I.copy()
        cv2.polylines(ImgDisp, [np.int32(Pts)], 1, ColorSpec)
        if(Disp is True):
            cv2.imshow(ImgTitle, ImgDisp)
            cv2.waitKey(WaitTime)
        return ImgDisp
    
