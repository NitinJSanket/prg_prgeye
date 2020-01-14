#!/usr/bin/env python


# Dependencies:
# opencv, do (pip install opencv-python)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)

import glob
import os
from termcolor import colored, cprint
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import random
import re
import h5py
import ImageUtils as iu


def Mat2HDF5(ReadPath):
    for dirs in tqdm(glob.glob(ReadPath + os.sep + '*' + '.mat')):
        Heatmap = sio.loadmat(PicklePath)['heatmap']
        print(dirs)
        input('q')
        Hf = h5py.File(dirs[:-4]+'.h5', 'w')
        Hf.create_dataset('heatmap', data=Heatmap)
        Hf.close()
        input('q')       

def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ReadPath', default='/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/dataset',
                        help='Base path of images, Default:/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/dataset')
    Parser.add_argument('--WritePath', default='/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/processed',
                        help='Base path of images, Default:/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/processed')
    
    Args = Parser.parse_args()
    ReadPath = Args.ReadPath
    WritePath = Args.WritePath
    
    if(not os.path.exists(WritePath)):
        cprint("WARNING: %s doesnt exist, Creating it."%WritePath, 'yellow')
        os.mkdir(WritePath)

    
if __name__ == '__main__':
    main()

    
