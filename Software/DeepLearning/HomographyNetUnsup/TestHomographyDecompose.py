#!/usr/bin/env python

import cv2
import numpy as np




def main():
	# Rotation Part
    # MaxRVal = np.array([2.0, 2.0, 2.0]) # In Degrees, Using 30 fps at 60 deg/s 
    # EulAng = 2*MaxRVal*([np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5])
    # R = Rot.from_euler('zyx', EulAng, degrees=True).as_dcm()
    R = np.eye(3)

    # Translation Part
    # MaxTVal = np.array([[0.16], [0.16], [0]]) # In m, Using 30 fps at 5 m/s
    # T = np.array(2*MaxTVal*([[np.random.rand() - 0.5],[np.random.rand() - 0.5],[np.random.rand() - 0.5]]))
    T = np.array([[1.0], [1.0], [1.0]])
    
    # Normal Part
    N = np.array([[0.0], [0.0], [1.0]]) # Keep this fixed 
    N = np.divide(N, np.linalg.norm(N))

    ImageSize = np.array([100.0, 100.0]);

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

    num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(H, K)
    
    print('Estimated')
    print(num)
    print(Rs)
    print(Ts)
    print(Ns)
    print(np.matmul(Ns[0].T, np.matmul(Rs[0].T, T)))
    print(np.matmul(Ns[1].T, np.matmul(Rs[1].T, T)))
    print(np.matmul(Ns[2].T, np.matmul(Rs[2].T, T)))
    print(np.matmul(Ns[3].T, np.matmul(Rs[3].T, T)))
    print('Actual')
    print(R)
    print(T)
    print(N)
    # print(H)
    
if __name__ == '__main__':
    main()

