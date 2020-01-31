# TODO: Test ComposeReducedH function

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
    AppendFlag = False
    if(len(np.shape(I)) == 3):
        I = I[np.newaxis, :, :, :] # Append Batch Dim
        AppendFlag = True
    ImageSize = np.shape(I)
    CenterX = ImageSize[1]/2
    CenterY = ImageSize[2]/2
    try:
        ICrop = I[:, int(np.ceil(CenterX-OutShape[0]/2)):int(np.ceil(CenterX+OutShape[0]/2)),\
                  int(np.ceil(CenterY-OutShape[1]/2)):int(np.ceil(CenterY+OutShape[1]/2)), :]
        if(AppendFlag): # Remove Batch Dim
            ICrop = np.squeeze(ICrop, axis=0)
    except:
        ICrop = None
    if (OutShape[0] > ImageSize[1]) or (OutShape[1] > ImageSize[2]):
        ICrop = None
        
    return ICrop

def CenterCropFactor(I, Factor):
    AppendFlag = False
    if(len(np.shape(I)) == 3):
        I = I[np.newaxis, :, :, :] # Append Batch Dim
        AppendFlag = True
    ImageSize = np.array(np.shape(I))
    CenterX = ImageSize[1]/2
    CenterY = ImageSize[2]/2
    OutShape = ImageSize - (np.mod(ImageSize,2**Factor))
    OutShape[3] = ImageSize[3]
    try:
        ICrop = I[:, int(np.ceil(CenterX-OutShape[1]/2)):int(np.ceil(CenterX+OutShape[1]/2)),\
                  int(np.ceil(CenterY-OutShape[2]/2)):int(np.ceil(CenterY+OutShape[2]/2)), :]
        if(AppendFlag): # Remove Batch Dim
            ICrop = np.squeeze(ICrop, axis=0)
    except:
        ICrop = None
        OutShape = None
    if (OutShape[1] > ImageSize[1]) or (OutShape[2] > ImageSize[2]):
        ICrop = None
        OutShape = None
    return (ICrop, OutShape)

def RandomCrop(I, OutShape):
    AppendFlag = False
    if(len(np.shape(I)) == 3):
        I = I[np.newaxis, :, :, :] # Append Batch Dim
        AppendFlag = True
    ImageSize = np.shape(I)
    try:
        RandX = random.randint(0, ImageSize[1]-OutShape[0])
        RandY = random.randint(0, ImageSize[2]-OutShape[1])
        ICrop = I[:, RandX:RandX+OutShape[0], RandY:RandY+OutShape[1], :]
        if(AppendFlag): # Remove Batch Dim
            ICrop = np.squeeze(ICrop, axis=0)
    except:
        ICrop = None
    return (ICrop)

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

class HomographyICTSN:
    def __init__(self, MaxParams = np.zeros((3,1)), TransformType = 'translation', MiniBatchSize = None):
        self.MaxParams = MaxParams
        self.TransformType = TransformType
        self.MiniBatchSize = MiniBatchSize

    def ComposedReducedHICSTN(self, Params = np.zeros((1,3))):
        if(not isinstance(self.TransformType, str)):
            print('ERROR: TransformType as to be a string')
            sys.exit(0)

        def ComposedReducedHICSTNSingle(TransformType = self.TransformType, Params = Params):
            H = np.eye(3)
            if(TransformType == 'pseudosimilarity'):
                H[0,0] = 1. + Params[0] # 1 + pc
                H[1,1] = 1. + Params[0] # 1 + pc
                H[0,2] = Params[1] # tx
                H[1,2] = Params[2] # ty
            elif(TransformType == 'similarity'):
                H[0,0] = 1. + Params[0] # 1 + pc
                H[1,1] = 1. + Params[0] # 1 + pc
                H[0,1] = Params[1] # ps
                H[1,0] = Params[1] # ps
                H[0,2] = Params[2] # tx
                H[1,2] = Params[3] # ty
            elif(TransformType == 'translation'):
                H[0,2] = Params[0] # tx
                H[1,2] = Params[1] # ty
            elif(TransformType == 'affine'):
                H[0,0] = 1. + Params[0] # 1 + p1
                H[0,1] = Params[0] # p2
                H[0,2] = Params[2] # p3
                H[1,0] = Params[3] # p4
                H[1,1] = 1. + Params[4] # 1 + p5
                H[1,2] = Params[5] # p6
            elif(TransformType == 'homography'):
                H[0,0] = 1. + Params[0] # 1 + p1
                H[0,1] = Params[0] # p2
                H[0,2] = Params[2] # p3
                H[1,0] = Params[3] # p4
                H[1,1] = 1. + Params[4] # 1 + p5
                H[1,2] = Params[5] # p6
                H[2,0] = Params[6] # p7
                H[2,1] = Params[7] # p8
            elif(TransformType == 'yaw'):
                print('ERROR: Not implemented yet')
                sys.exit(0)
            elif(TransformType == 'scale'):
                print('ERROR: Not implemented yet')
                sys.exit(0)
            return H[np.newaxis, :, :]

        if(self.MiniBatchSize is None):
            # Runs once
            H = ComposedReducedHICSTNSingle(TransformType = self.TransformType, Params = np.squeeze(Params))
            return H
        else:
            for count in range(self.MiniBatchSize):
                H = ComposedReducedHICSTNSingle(TransformType = self.TransformType[count], Params = Params[count])
                if(count == 0):
                    HAppend = H
                else:
                    HAppend = np.append(HAppend, H, axis=0)
            return HAppend

    def GetRandParamsICSTN(self, MaxParams = None):
        if MaxParams is not None:
            # Overwrite value
            self.MaxParams = MaxParams
        def RandSample(Max, Min = None):
            if(Min is None):
                Min = -Max
            return (Max - Min)*(np.random.rand()) + Min
        
        def GetRandParamsICSTNSingle(self, TransformType = 'yaw'):
            # TODO: Cleanup by defining self.warpDim
            if(TransformType == 'pseudosimilarity'):
                Params = np.array([RandSample(self.MaxParams[0]), RandSample(self.MaxParams[1]), RandSample(self.MaxParams[2])])
            elif(TransformType == 'similarity'):
                Params = np.array([RandSample(self.MaxParams[0]), RandSample(self.MaxParams[1]), RandSample(self.MaxParams[2]),\
                                   RandSample(self.MaxParams[3])])
            elif(TransformType == 'translation'):
                Params = np.array([RandSample(self.MaxParams[0]), RandSample(self.MaxParams[1])])
            elif(TransformType == 'affine'):
               Params = np.array([RandSample(self.MaxParams[0]), RandSample(self.MaxParams[1]), RandSample(self.MaxParams[2]),\
                                  RandSample(self.MaxParams[3]), RandSample(self.MaxParams[4]), RandSample(self.MaxParams[5])])
            elif(TransformType == 'homography'):
                Params = np.array([RandSample(self.MaxParams[0]), RandSample(self.MaxParams[1]), RandSample(self.MaxParams[2]),\
                                   RandSample(self.MaxParams[3]), RandSample(self.MaxParams[4]), RandSample(self.MaxParams[5]),\
                                   RandSample(self.MaxParams[6]), RandSample(self.MaxParams[7])])
            elif(TransformType == 'yaw'):
                print('ERROR: Not implemented yet')
                sys.exit(0)
            elif(TransformType == 'scale'):
                print('ERROR: Not implemented yet')
                sys.exit(0)
            return Params[np.newaxis,:]
        
        if(self.MiniBatchSize is None):
            # Runs once
            Params = GetRandParamsICSTNSingle(self, TransformType = self.TransformType)
            return Params
        else:
            for count in range(self.MiniBatchSize):
                Params = GetRandParamsICSTNSingle(self, TransformType = self.ransformType[count])
                if(count == 0):
                    ParamsAppend = Params
                else:
                    ParamsAppend = np.append(ParamsAppend, Params, axis=0)
            return ParamsAppend        
        

    def GetRandReducedHICSTN(self, MaxParams = None):
        Params = self.GetRandParamsICSTN(MaxParams = MaxParams) # Get Params Batch
        H = self.ComposedReducedHICSTN(Params = Params)
        return H, Params



class Homography:
    def __init__(self, ImageSize=[128., 128., 3.], MaxR = np.zeros((1,3)), MaxT = np.array([[0.25], [0.25], [0.25]]), MaxYaw = 45.0, MaxMinScale = np.array([0.7, 1.4])):
        self.ImageSize = ImageSize
        self.ScaleMtrx = np.eye(3) # Scales from [-1, 1] ImageCoordinates to Actual Image Coordinates
        self.ScaleMtrx[0,0] = ImageSize[1]/2
        self.ScaleMtrx[0,2] = ImageSize[1]/2
        self.ScaleMtrx[1,1] = ImageSize[0]/2
        self.ScaleMtrx[1,2] = ImageSize[0]/2
        self.MaxR = MaxR # +- Degrees Euler Angles ZYX
        self.MaxT = MaxT # +- T in f units or px. if using 2D
        self.MaxYaw = MaxYaw # +- Degrees
        self.MaxMinScale = MaxMinScale # [Min, Max]
        
    def ComposeHFromRTN(self, R = np.eye(3), T = np.zeros((3, 1)), N = np.array([[0.], [0.], [1.]]), ScaleToPx = False):
        H = np.add(R, np.matmul(T, N.T)) # R + TN'
        H = np.divide(H, H[2,2]) # Nornalize by making last element 1
        if(ScaleToPx):
            H =  np.matmul(self.ScaleMtrx, np.matmul(H, np.linalg.inv(self.ScaleMtrx))) # Scale to bring to Image Coordinates
           
        return H

    def ComposeReducedH(self, TransformType = ['Yaw', 'Scale', 'T2D'], T2D = np.zeros((2, 1)), Yaw = 0.0, Scale =  np.ones((2, 1)), Shear = np.zeros((2, 1)), ScaleToPx = False):
        # T2D is in px.
        # Yaw is in degrees
        # Scale is percentage of f, 1.0 gives original scale
        # Transformation order is always Scale -> Yaw -> Shear -> Translation
        # Notes from here: https://courses.cs.washington.edu/courses/csep576/11sp/pdf/Transformations.pdf
        def HFromYaw(Yaw):
            Yawr = np.radians(Yaw, ScaleToPx)
            cosYaw = np.cos(Yawr)
            sinYaw = np.sin(Yawr)
            HNow = np.eye(3)
            HNow[0,0] = cosYaw
            HNow[0,1] = -sinYaw
            HNow[1,0] = sinYaw
            HNow[1,1] = cosYaw
            if(ScaleToPx):
                HNow =  np.matmul(self.ScaleMtrx, np.matmul(HNow, np.linalg.inv(self.ScaleMtrx))) # Scale to bring to Image Coordinates
            return HNow
       
        def HFromScale(Scale, ScaleToPx):
            HNow = np.eye(3)
            HNow[0,0] = Scale[0]
            HNow[1,1] = Scale[1]
            if(ScaleToPx):
                HNow =  np.matmul(self.ScaleMtrx, np.matmul(HNow, np.linalg.inv(self.ScaleMtrx))) # Scale to bring to Image Coordinates
            return HNow

        def HFromShear(Shear, ScaleToPx):
            HNow = np.eye(3)
            HNow[0,1] = Shear[0]
            HNow[1,0] = Shear[1]
            if(ScaleToPx):
                HNow =  np.matmul(self.ScaleMtrx, np.matmul(HNow, np.linalg.inv(self.ScaleMtrx))) # Scale to bring to Image Coordinates
            return HNow

        def HFromTranslation2D(T2D, ScaleToPx):
            HNow = np.eye(3)
            HNow[0,2] = T2D[0]
            HNow[1,2] = T2D[1]
            if(ScaleToPx):
                HNow =  np.matmul(self.ScaleMtrx, np.matmul(HNow, np.linalg.inv(self.ScaleMtrx))) # Scale to bring to Image Coordinates
            return HNow


        # If TranformType is not list, make it into a list
        if(not isinstance(TransformType, list)):
            TransformType = list(TransformType)
            
        # Any combination composition is possible, list order does not matter
        H = np.eye(3)
        for count in TransformType:
            if 'Scale' in TransformType:
                HNow = HFromScale(Scale, ScaleToPx)
                H = np.matmul(H, HNow)
            if 'Yaw' in TransformType:
                HNow = HFromYaw(Yaw, ScaleToPx)
                H = np.matmul(H, HNow)
            if 'Shear' in TransformType:
                HNow = HFromShear(Shear, ScaleToPx)
                H = np.matmul(H, HNow)
            if 'T2D' in TransformType:
                HNow = HFromTranslation2D(T2D, ScaleToPx)
                H = np.matmul(H, HNow)            
        return H 
        
    def DecomposeHToRTN(self):
        # retval, rotations, translations, normals  =  cv.decomposeHomographyMat(H, K[, rotations[, translations[, normals]]])
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

    def GetRandR(self, MaxR = None, EulOrder='zyx'):
        # MaxR is given in Degrees
        if MaxR is not None:
            # Overwrite value
            self.MaxR = np.radians(MaxR)
        # Generate random value of euler angles
        EulAng = 2*self.MaxR*([np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5])
        R = Rot.from_euler(EulOrder, EulAng, degrees=True).as_dcm()
        return R, EulAng

    def GetRandT(self, MaxT = None, Flag2D = False):
        # MaxT is given in Percentage of focal length (0.1 means 0.1f)
        if MaxT is not None:
            # Overwrite value
            self.MaxT = MaxT
        # Generate random value of translation
        if(not Flag2D):
            T = np.array(4*self.MaxT*([[np.random.rand() - 0.5],[np.random.rand() - 0.5],[np.random.rand() - 0.5]]))  # 2x2, 2 for ImageSize/2 and 2 for rand scaling
        else:
            self.MaxT = self.MaxT[0:2] # Extract first two elements if three are given
            T = np.array(4*self.MaxT*([[np.random.rand() - 0.5],[np.random.rand() - 0.5]])) # 2x2, 2 for ImageSize/2 and 2 for rand scaling
        return T

    def GetRandYaw(self, MaxYaw = None):
        if MaxYaw is not None:
            # Overwrite value
            self.MaxYaw = MaxYaw
        return 2*self.MaxYaw*(np.random.rand() - 0.5)

    def GetRandScale(self, MaxMinScale = None, Uniform = True):
        if MaxMinScale is not None:
            # Overwrite value
            self.MaxMinScale = MaxMinScale
        if Uniform:
            Scale = (self.MaxMinScale[1] - self.MaxMinScale[0])*np.random.rand() + self.MaxMinScale[0]
            Scale = np.tile(Scale, [2,1])
        else:
            Scale = [[(self.MaxMinScale[1] - self.MaxMinScale[0])*np.random.rand() + self.MaxMinScale[0]],\
                     [(self.MaxMinScale[1] - self.MaxMinScale[0])*np.random.rand() + self.MaxMinScale[0]]]
        return Scale

    def GetRandReducedH(self, TransformType = ['Yaw', 'Scale', 'T2D'], MaxT = None, MaxYaw = None, MaxMinScale = None, ScaleToPx = False):
        if(not isinstance(TransformType, list)):
            TransformType = list(TransformType)

        T2D = None
        Yaw = None
        Scale = None
        Shear = None
        for count in TransformType:
            if 'Scale' in TransformType:
                Scale = self.GetRandScale(MaxMinScale, True)       
            if 'Yaw' in TransformType:
                Yaw = self.GetRandYaw(MaxYaw)        
            if 'Shear' in TransformType:
               Shear = None # Not implemented yet!
            if 'T2D' in TransformType:
               T2D = self.GetRandT(MaxT, True)
            
        Compositions = [Scale, Yaw, Shear, T2D]
        H = self.ComposeReducedH(TransformType, T2D = T2D, Yaw = Yaw, Scale = Scale, ScaleToPx = ScaleToPx)
        return H, Compositions
                            
        
    

            
        
        
    
