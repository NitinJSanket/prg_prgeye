#!/usr/bin/env python
# Code adapted from: https://akshaysin.github.io/fourier_transform.html#.XSYBbnVKhhF

import cv2
import numpy as np
import Misc.MiscUtils as mu
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as Rot
import Misc.ImageUtils as iu

def main():
    BasePath = '/home/nitin/Datasets/MSCOCO/train2014/COCO_train2014_000000581882.jpg'
    I = cv2.imread(BasePath)
    ImageSize = np.array([300, 300, 3])
    I = iu.RandomCrop(I, ImageSize)
    # ImageSize = np.shape(I)
    
    # Rotation Part
    MaxRVal = np.array([2.0, 2.0, 2.0]) # In Degrees, Using 30 fps at 60 deg/s 
    EulAng = 2*MaxRVal*([np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5])
    R = Rot.from_euler('zyx', EulAng, degrees=True).as_dcm()

    # Translation Part
    MaxTVal = np.array([[0.166], [0.166], [0.166]]) # In m, Using 30 fps at 5 m/s
    T = np.array(2*MaxTVal*([[np.random.rand() - 0.5],[np.random.rand() - 0.5],[np.random.rand() - 0.5]]))
    
    # Normal Part
    N = np.array([[0.0], [0.0], [1.0]]) # Keep this fixed 
    N = np.divide(N, np.linalg.norm(N))

    # Camera Matrix
    K = np.eye(3)
    K[0,0] = 400.0
    K[1,1] = 400.0
    K[0,2] = ImageSize[0]/2
    K[1,2] = ImageSize[1]/2
    
    # Compose Homography
    H = np.add(R, np.matmul(T, N.T))
    H = np.divide(H, H[2,2])
    H = np.matmul(K, np.matmul(H, np.linalg.inv(K)))
    
    # Get Inverse Homography
    HInv = np.linalg.inv(H)
    
    IWarp = cv2.warpPerspective(I, H, (ImageSize[1],ImageSize[0]))
    # Display Images
    cv2.imshow('I', I)
    cv2.imshow('IWarp', IWarp)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
