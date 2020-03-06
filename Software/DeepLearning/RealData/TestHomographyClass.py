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

    # cv2.imshow('a', I)
    # cv2.waitKey(0)
    

    HObj = iu.Homography(ImageSize = ImageSize, MaxT = np.array([[0.025], [0.025], [0.025]]), MaxYaw = 1.2, MaxMinScale = np.array([0.97, 1.03]))
    H = HObj.ComposeReducedH(TransformType = ['Scale', 'T2D'], T2D = np.array([[0.1],[0.1]]), Scale = np.array([[0.5],[0.5]]), ScaleToPx = True)

    # print(H)

    # H, Compositions = HObj.GetRandReducedH(TransformType = ['Yaw', 'Scale', 'T2D'], ScaleToPx = False) # ['Yaw', 'Scale', 'T2D']
    # print(H)
    # print(Compositions)

    # I2 = warp2.transformImageNP(opt, I, H)

    opt = warp2.Options(PatchSize=ImageSize, MiniBatchSize=1, warpType= ['pseudosimilarity'])

    ## OpenCV Warping
    # scale = 0.5 # 1.0/(pc+1)
    # tx = 0.0
    # ty = 0.0
    # Hs = np.eye(3)
    # Hs[0,0] = scale
    # Hs[1,1] = scale
    # Ht = np.eye(3)
    # Ht[0,2] = scale*tx
    # Ht[1,2] = scale*ty

    

    # Hcv = np.matmul(Hs, Ht)
    A = opt.refMtrx
    # HCorr = np.eye(3)
    # HCorr[0,0] = scale
    # HCorr[0,2] = tx/scale
    # HCorr[1,1] = scale
    # HCorr[1,2] = ty/scale

    # H = np.eye(3)
    # pc = 1.0
    # tx = 0.0
    # ty = 0.0
    # H[0,0] = 1. + pc
    # H[1,1] = 1. + pc
    # H[0,2] = tx
    # H[1,2] = ty

    # Hcv = H.copy()

    
    # A = np.eye(3)
    # A[0,0] = ImageSize[1]/2 
    # A[0,2] = ImageSize[1]/2 
    # A[1,1] = ImageSize[0]/2 
    # A[1,2] = ImageSize[0]/2 
    # AInv = np.linalg.inv(A)
    # A2 = A.copy()
    # A2[0,0] = 1.
    # A2[1,1] = 1.
    # Hcv = np.matmul(np.matmul(A, Hcv), AInv)
    # Hcv = np.matmul(np.matmul(A, Hcv), A2)
    # print(opt.refMtrx)



    ## TF Warping
    

    ## Test fitting H
    PatchSize = ImageSize
    canon4pts = np.array([[-1,-1],[-1,1],[1,1],[1,-1]],dtype=np.float32)
    image4pts = np.array([[0,0],[0,PatchSize[1]-1],[PatchSize[0]-1,PatchSize[1]-1],[PatchSize[0]-1,0]],dtype=np.float32)
    refMtrx = warp2.fit(Xsrc=canon4pts, Xdst=image4pts)


    AInv = np.linalg.inv(refMtrx)

    # Hcv = H.copy()
    # Hcv[0,0] = H[0,0] - 1.
    # Hcv[1,1] = H[1,1] - 1.
    # Hcv = np.matmul(np.matmul(np.linalg.inv(refMtrx), Hcv), refMtrx)    # np.matmul(np.matmul(A,np.matmul(refMtrx, H)), AInv)
    
    
    # Hcv = np.matmul(np.matmul(refMtrx, Hcv), np.linalg.inv(refMtrx))


    # Hcv = np.eye(3)
    # scale = 1.0 # pct. of W
    # tx = 10.0 # px.
    # ty = 10.0 # px.
    # Hcv[0,0] = scale
    # Hcv[1,1] = scale
    # Hcv[0,2] = tx
    # Hcv[1,2] = ty

    # A = np.eye(3)
    # A[0,2] = ImageSize[1]/2
    # A[1,2] = ImageSize[0]/2
    # AInv = np.linalg.inv(A)

    
    # Hcv = np.matmul(np.matmul(A, Hcv), AInv)
    # Hcv = np.matmul(A, Hcv)

    H = np.eye(3)
    pc = 1.0
    tx = 1.0
    ty = 1.0
    H[0,0] = 1. + pc
    H[1,1] = 1. + pc
    H[0,2] = tx
    H[1,2] = ty
    
    I2 = np.squeeze(warp2.transformImageNP(opt, I[np.newaxis,:,:,:], H[np.newaxis,:,:]))
    # I2 = cv2.warpPerspective(I, Hcv, (ImageSize[1], ImageSize[0]))
    cv2.imshow('I, WarpedICV', np.hstack((I, I2)))
    cv2.waitKey(0)


    ## TF
    # H = np.eye(3)
    # pc = 0.0
    # tx = 10./150.
    # ty = 10./150.
    # H[0,0] = 1. + pc
    # H[1,1] = 1. + pc
    # H[0,2] = tx
    # H[1,2] = ty

    ITensor = tf.convert_to_tensor(np.float32(I), dtype='float')
    HTensor = tf.convert_to_tensor(np.float32(H), dtype='float')
    
    WarpI1PatchIdeal, transMtrx = warp2.transformImage(opt, ITensor, HTensor)

    with tf.Session() as sess:
         WarpI1IdealRet = WarpI1PatchIdeal.eval()
         transMtrxRet = transMtrx.eval()

    # print(transMtrxRet[0])

    # cv2.imshow('b', np.uint8(transMtrxRet[0]))
    # cv2.waitKey(0)

    
    WarpI1IdealRet = np.uint8(WarpI1IdealRet[0])
    cv2.imshow('I, WarpedITF', np.hstack((I, WarpI1IdealRet)))
    cv2.imshow('Diff', np.abs(I2-WarpI1IdealRet))
    cv2.waitKey(0)
    
    # H = np.matmul(HCorr, Hcv)
    

    # ScaleMtrx = np.eye(3) # Scales from [-1, 1] ImageCoordinates to Actual Image Coordinates
    # ScaleMtrx[0,0] = ImageSize[1]/2
    # ScaleMtrx[0,2] = ImageSize[1]/2
    # ScaleMtrx[1,1] = ImageSize[0]/2
    # ScaleMtrx[1,2] = ImageSize[0]/2

    # ScaleMtrx = opt.refMtrx
    # scalecv = 1/((ImageSize[1]/2)*(1+pc)) # 1/((ImageSize[1]/2)*(1+pc))
    # txcv = -(tx/scalecv) + 1 # - 1 # (tx*scalecv) # + 1 # - 1
    # tycv = -(ty/scalecv) + 1# - 1 # (ty*scalecv) # + 1 # - 1
    # H1 = np.eye(3)
    # H1[0,0] = scalecv
    # H1[1,1] = scalecv
    # H1[0,2] = scalecv*txcv
    # H1[1,2] = scalecv*tycv
    # # H1 = np.matmul(ScaleMtrx, np.matmul(H1, np.linalg.inv(ScaleMtrx)))
    # AInv = np.linalg.inv(ScaleMtrx)
    # A = ScaleMtrx.copy()
    # A2 = ScaleMtrx.copy()
    # A2[0,0] = 1.
    # A2[1,1] = 1.
    # H2 = np.matmul(A, np.matmul(H1, np.linalg.inv(A2))) # np.matmul(A, np.matmul(H1, np.linalg.inv(A)))# np.matmul(np.matmul(A, H1), AInv)

    # print(np.matmul(A, H))
    # print(H2)

    # H2 = np.eye(3)
    # scale1 = 1/1.5 # goes from 0.0 to 2.0, larger number in dr zooms out
    # tx1 = 0.
    # ty1 = 0.
    # H2[0,0] = scale1
    # H2[1,1] = scale1
    # H2[0,2] = scale1*tx1
    # H2[1,2] = scale1*ty1

    # H2 = np.matmul(np.matmul(A, H2), np.linalg.inv(A))
    
    # I2 = cv2.warpPerspective(I, np.matmul(A, H), (ImageSize[1], ImageSize[0]))

    # cv2.imshow('I, WarpedI', np.hstack((I, I2)))
    # cv2.waitKey(0)
    
    
    # ScaleMtrx = np.eye(3) # Scales from [-1, 1] ImageCoordinates to Actual Image Coordinates
    # ScaleMtrx[0,0] = ImageSize[1]/2
    # ScaleMtrx[0,2] = ImageSize[1]/2
    # ScaleMtrx[1,1] = ImageSize[0]/2
    # ScaleMtrx[1,2] = ImageSize[0]/2
    # print(np.matlmul(ScaleMtrx, np.linalg.inv(ScaleMtrx)))
    # H =  np.matmul(ScaleMtrx, np.matmul(H, np.linalg.inv(ScaleMtrx)))
    # H = np.matmul(ScaleMtrx, H)
    # print(H)

    # WarpedI = cv2.warpPerspective(I, H, (ImageSize[1], ImageSize[0]))

    # Crop in center for PatchSize
    # PatchSize = np.array([128, 128, 3])
    # I1 = iu.CenterCrop(I, PatchSize)
    # I2 = iu.CenterCrop(WarpedI, PatchSize)
    

    # cv2.imshow('I, WarpedI', np.hstack((I, WarpedI)))
    # cv2.imshow('I1, I2', np.hstack((I1, I2)))
    

    
if __name__=="__main__":
    main()
