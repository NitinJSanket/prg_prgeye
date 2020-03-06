#!/usr/bin/env python

import Misc.ImageUtils as iu
import cv2
import numpy as np
import Misc.warpICSTN2 as warp2
import tensorflow as tf
import Misc.MiscUtils as mu
import Misc.TFUtils as tu


def RandSimilarityPerturbation(I1, HObj, PatchSize, ImageSize=None, Vis=False):
    if(ImageSize is None):
        ImageSize = np.array(np.shape(I1))

    H, Params = HObj.GetRandReducedHICSTN(TransformType = 'psuedosimilarity')
    opt = warp2.Options(PatchSize=ImageSize, MiniBatchSize=1, warpType= ['pseudosimilarity'])

    # Numpy based Warping
    I2 = warp2.transformImageNP(opt, I1[np.newaxis,:,:,:], H[np.newaxis,:,:])[0]

    # Crop in center for PatchSize
    P1 = iu.CenterCrop(I1, PatchSize)
    P2 = iu.CenterCrop(I2, PatchSize)
    
    if(Vis is True):
        cv2.imshow('I1, I2', np.hstack((I1, I2)))
        cv2.imshow('I1, I2', np.hstack((P1, P2)))
        cv2.waitKey(0)
    
 
    # P1 is I1 cropped to patch Size
    # P2 is I1 Crop Warped (I2 Crop)
    # H is Homography
    # Params is the stuff H is made from 
    return I1, I2, P1, P2, H, Params


def main():
    # Set GPUDevice
    GPUDevice = 1
    tu.SetGPU(GPUDevice)
    
    # Test Image Warping
    BasePath = '/home/nitin/Datasets/MSCOCO/train2014/COCO_train2014_000000484344.jpg'
    I = cv2.imread(BasePath)
    ImageSize = np.array([300, 300, 3])
    MiniBatchSize = 32
    I = iu.CenterCrop(I, ImageSize)
    I = np.zeros_like(I)
    I[100:200, 100:200, :] = 255.0
    ITile = np.tile(I[np.newaxis,:,:,:], (MiniBatchSize,1,1,1))

    
    MaxParams = np.array([0.5, 0.2, 0.2, 0.2])
    # HObj = iu.Homography(ImageSize = ImageSize, MaxT = np.array([[0.025], [0.025], [0.025]]), MaxYaw = 1.2, MaxMinScale = np.array([0.97, 1.03]), MaxParams = MaxParams)
    HObj = iu.HomographyICTSN(MaxParams = MaxParams)
    # H = HObj.ComposeReducedH(TransformType = ['Scale', 'T2D'], T2D = np.array([[0.1],[0.1]]), Scale = np.array([[0.5],[0.5]]), ScaleToPx = True)
    # H = HObj.ComposedReducedHICSTN(TransformType = 'psuedosimilarity', Params = MaxParams)
    Timer1 = mu.tic()
    H, Params = HObj.GetRandReducedHICSTN(TransformType = 'similarity', MaxParams = MaxParams, MiniBatchSize = MiniBatchSize)
    print(mu.toc(Timer1))

    opt = warp2.Options(PatchSize=ImageSize, MiniBatchSize=MiniBatchSize, warpType= ['similarity'])

    ## OpenCV Warping
    for count in range(1):
        Timer1 = mu.tic()
        I2 = warp2.transformImageNP(opt, ITile, H)
        print(mu.toc(Timer1))
    print('-----------------')
    I2 = I2[0]# np.uint8(np.round(I2[0]))# np.uint8(np.ceil(np.clip(I2[0], 0.0, 255.0)))# np.uint8(np.round(I2[0])) # np.uint8(mu.remap(I2[0], 0.0, 255.0, np.amin(I2), np.amax(I2)))
    # cv2.imshow('I, WarpedICV', np.hstack((I, I2)))
    # cv2.waitKey(0)

    # opt = warp2.Options(PatchSize=ImageSize, MiniBatchSize=MiniBatchSize, warpType= ['pseudosimilarity'])
    ## TF Warping
    ITensor = tf.convert_to_tensor(np.float32(ITile), dtype='float')
    HTensor = tf.convert_to_tensor(np.float32(H), dtype='float')
    
    WarpI1PatchIdeal = warp2.transformImage(opt, ITensor, HTensor)

    with tf.Session() as sess:
        for count in range(10):
            Timer1 = mu.tic()
            WarpI1IdealRet = WarpI1PatchIdeal.eval()
            print(mu.toc(Timer1))

            
    # print(transMtrxRet[0], transMtrxRetcv[0])
    # print(np.any(transMtrxRet[0] - transMtrxRetcv[0]))
    WarpI1IdealRet = np.uint8(WarpI1IdealRet[0])
    cv2.imshow('I, WarpedITF', np.hstack((I, WarpI1IdealRet)))
    cv2.imshow('Diff', np.abs(I2-WarpI1IdealRet))
    print('Diff in values: ', np.sum(np.abs(I2-WarpI1IdealRet)>1.0))
    cv2.waitKey(0)

    
if __name__=="__main__":
    main()
