#!/usr/bin/env python

import Misc.ImageUtils as iu
import cv2

def main():
    HObj = iu.Homography()
    H = HObj.GetRandReducedH() 
    print(H)

    # TODO: Test Image warping
    

    
if __name__=="__main__":
    main()
