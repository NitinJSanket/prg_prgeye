#!/usr/bin/env python

import Misc.ImageUtils as iu
import cv2
import numpy as np
import Misc.warpICSTN2 as warp2
import tensorflow as tf
import Misc.MiscUtils as mu


def main():
    # Test Image Warping
    BasePath = '/home/nitin/Datasets/MSCOCO/train2014/COCO_train2014_000000484344.jpg'
    I = cv2.imread(BasePath)
    ImageSize = np.array([300, 300, 3])
    I = iu.CenterCrop(I, ImageSize)
    I = np.zeros_like(I)
    I[100:200, 100:200, :] = 255.0

    MaxParams = np.array([0.5, 0.2, 0.2])
    # HObj = iu.Homography(ImageSize = ImageSize, MaxT = np.array([[0.025], [0.025], [0.025]]), MaxYaw = 1.2, MaxMinScale = np.array([0.97, 1.03]), MaxParams = MaxParams)
    HObj = iu.HomographyICTSN(MaxParams = MaxParams)
    # H = HObj.ComposeReducedH(TransformType = ['Scale', 'T2D'], T2D = np.array([[0.1],[0.1]]), Scale = np.array([[0.5],[0.5]]), ScaleToPx = True)
    # H = HObj.ComposedReducedHICSTN(TransformType = 'psuedosimilarity', Params = MaxParams)
    H = HObj.GetRandReducedHICSTN(TransformType = 'psuedosimilarity', MaxParams = MaxParams)

    opt = warp2.Options(PatchSize=ImageSize, MiniBatchSize=1, warpType= ['pseudosimilarity'])

    ## OpenCV Warping
    I2 = warp2.transformImageNP(opt, I[np.newaxis,:,:,:], H[np.newaxis,:,:])
    I2 = I2[0]# np.uint8(np.round(I2[0]))# np.uint8(np.ceil(np.clip(I2[0], 0.0, 255.0)))# np.uint8(np.round(I2[0])) # np.uint8(mu.remap(I2[0], 0.0, 255.0, np.amin(I2), np.amax(I2)))
    cv2.imshow('I, WarpedICV', np.hstack((I, I2)))
    cv2.waitKey(0)

    ## TF Warping
    ITensor = tf.convert_to_tensor(np.float32(I), dtype='float')
    HTensor = tf.convert_to_tensor(np.float32(H), dtype='float')
    
    WarpI1PatchIdeal = warp2.transformImage(opt, ITensor, HTensor)

    with tf.Session() as sess:
         WarpI1IdealRet = WarpI1PatchIdeal.eval()

    # print(transMtrxRet[0], transMtrxRetcv[0])
    # print(np.any(transMtrxRet[0] - transMtrxRetcv[0]))
    WarpI1IdealRet = np.uint8(WarpI1IdealRet[0])
    cv2.imshow('I, WarpedITF', np.hstack((I, WarpI1IdealRet)))
    cv2.imshow('Diff', np.abs(I2-WarpI1IdealRet))
    print('Diff in values: ', np.sum(np.abs(I2-WarpI1IdealRet)>1.0))
    cv2.waitKey(0)

    
if __name__=="__main__":
    main()
