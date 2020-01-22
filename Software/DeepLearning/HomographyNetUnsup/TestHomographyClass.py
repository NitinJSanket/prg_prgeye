#!/usr/bin/env python

import Misc.ImageUtils as iu
import cv2
import numpy as np

def main():
    

    
    # test Image Warping
    BasePath = '/home/nitin/Datasets/MSCOCO/train2014/COCO_train2014_000000484344.jpg'
    I = cv2.imread(BasePath)
    ImageSize = np.shape(I) 

    HObj = iu.Homography(MaxT = np.array([[0.2], [0.2], [0.2]]), MaxYaw = 30.0)
    H = HObj.GetRandReducedH(TransformType = ['T2D', 'Yaw', 'Scale'], ScaleToPx = True) # ['Yaw', 'Scale', 'T2D']
    print(H)

    
    # ScaleMtrx = np.eye(3) # Scales from [-1, 1] ImageCoordinates to Actual Image Coordinates
    # ScaleMtrx[0,0] = ImageSize[0]/2
    # ScaleMtrx[0,2] = ImageSize[0]/2
    # ScaleMtrx[1,1] = ImageSize[1]/2
    # ScaleMtrx[1,2] = ImageSize[1]/2
    # H =  np.matmul(ScaleMtrx, np.matmul(H, np.linalg.inv(ScaleMtrx)))
    

    WarpedI = cv2.warpPerspective(I, H, (ImageSize[1],ImageSize[0]))

    cv2.imshow('I, WarpedI', np.hstack((I, WarpedI)))
    cv2.waitKey(0)

    
if __name__=="__main__":
    main()
