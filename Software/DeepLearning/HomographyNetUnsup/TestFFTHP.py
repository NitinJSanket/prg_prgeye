#!/usr/bin/env python
# Code adapted from: https://akshaysin.github.io/fourier_transform.html#.XSYBbnVKhhF

import cv2
import numpy as np
import Misc.MiscUtils as mu

def main():
    BasePath = '/home/nitin/Datasets/MSCOCO/train2014/COCO_train2014_000000581882.jpg'
    I = cv2.imread(BasePath, 0)
    F = cv2.dft(np.float32(I), flags=cv2.DFT_COMPLEX_OUTPUT)
    FShift = np.fft.fftshift(F)
    Mag = 20 * np.log(cv2.magnitude(FShift[:, :, 0], FShift[:, :, 1]))
       
    # Circular HPF mask, center circle is 0, remaining all ones
    Rows, Cols = I.shape
    Mask = np.ones((Rows, Cols, 2), np.uint8)
    Radius = 10
    Center = [int(Rows / 2), int(Cols / 2)]
    x, y = np.ogrid[:Rows, :Cols]
    MaskArea = (x - Center[0]) ** 2 + (y - Center[1]) ** 2 <= Radius**2
    Mask[MaskArea] = 0

    # Filter by Masking FFT Spectrum
    FShiftFilt = np.multiply(FShift, Mask)
    FFilt = np.fft.ifftshift(FShiftFilt)
    IFilt = cv2.idft(FFilt)
    IFilt = cv2.magnitude(IFilt[:, :, 0], IFilt[:, :, 1])

    # Display Images
    cv2.imshow('I', I)
    cv2.imshow('FMag', np.uint8(mu.remap(Mag, 0.0, 255.0, np.amin(Mag), np.amax(Mag))))
    cv2.imshow('Mask', np.uint8(Mask[:,:,0]*255.0))
    cv2.imshow('IFilt', np.uint8(mu.remap(IFilt, 0.0, 255.0, np.amin(IFilt), np.amax(IFilt))))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

