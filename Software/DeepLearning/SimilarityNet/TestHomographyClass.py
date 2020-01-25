#!/usr/bin/env python

import Misc.ImageUtils as iu
import cv2
import numpy as np

def main():
    # Test Image Warping
    BasePath = '/home/nitin/Datasets/MSCOCO/train2014/COCO_train2014_000000484344.jpg'
    I = cv2.imread(BasePath)
    ImageSize = np.array([300, 300, 3])
    I = iu.CenterCrop(I, ImageSize)

    HObj = iu.Homography(ImageSize = ImageSize, MaxT = np.array([[0.025], [0.025], [0.025]]), MaxYaw = 1.2, MaxMinScale = np.array([0.97, 1.03]))
    H, Compositions = HObj.GetRandReducedH(TransformType = ['Yaw', 'Scale', 'T2D'], ScaleToPx = True) # ['Yaw', 'Scale', 'T2D']
    print(H)
    print(Compositions)

    
    # ScaleMtrx = np.eye(3) # Scales from [-1, 1] ImageCoordinates to Actual Image Coordinates
    # ScaleMtrx[0,0] = ImageSize[1]/2
    # ScaleMtrx[0,2] = ImageSize[1]/2
    # ScaleMtrx[1,1] = ImageSize[0]/2
    # ScaleMtrx[1,2] = ImageSize[0]/2
    # print(np.matlmul(ScaleMtrx, np.linalg.inv(ScaleMtrx)))
    # H =  np.matmul(ScaleMtrx, np.matmul(H, np.linalg.inv(ScaleMtrx)))
    # H = np.matmul(ScaleMtrx, H)
    # print(H)

    WarpedI = cv2.warpPerspective(I, H, (ImageSize[1], ImageSize[0]))

    # Crop in center for PatchSize
    PatchSize = np.array([128, 128, 3])
    I1 = iu.CenterCrop(I, PatchSize)
    I2 = iu.CenterCrop(WarpedI, PatchSize)
    

    cv2.imshow('I, WarpedI', np.hstack((I, WarpedI)))
    cv2.imshow('I1, I2', np.hstack((I1, I2)))
    cv2.waitKey(0)

    
if __name__=="__main__":
    main()
