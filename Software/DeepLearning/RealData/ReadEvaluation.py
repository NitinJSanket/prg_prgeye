#!/usr/bin/env python


import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

def main(ReadPath):
    SkipLines = 8
    Count = 0
    File = open(ReadPath, 'r')

    Delimiters = "\t", "\n"
    RegexPattern = '|'.join(map(re.escape, Delimiters))
    ErrorScalePxPred = []
    ErrorScalePxIdentity = []    
    ErrorTransPxPred = []
    ErrorTransPxIdentity = []
    
    for Count, Line in enumerate(File):
        if(Count > SkipLines):
            LineSplit = re.split(RegexPattern, Line)
            ErrorScalePxPred.append(float(LineSplit[3].strip('[]')))
            ErrorScalePxIdentity.append(float(LineSplit[4].strip('[]')))
            ErrorTransPxPred.append(float(LineSplit[5].strip('[]')))
            ErrorTransPxIdentity.append(float(LineSplit[6].strip('[]')))

    # print('Scale Pred Mean {} +- {} Px'.format(np.mean(ErrorScalePxPred), np.std(ErrorScalePxPred)))
    # print('Scale Identity Mean {} +- {} Px'.format(np.mean(ErrorScalePxIdentity), np.std(ErrorScalePxIdentity)))
    # print('Trans Pred Mean {} +- {} Px'.format(np.mean(ErrorTransPxPred), np.std(ErrorTransPxPred)))
    # print('Trans Identity Mean {} +- {} Px'.format(np.mean(ErrorTransPxIdentity), np.std(ErrorTransPxIdentity)))
    print('Scale Pred Median {} Px'.format(np.median(ErrorScalePxPred)))
    print('Scale Identity Median {} Px'.format(np.median(ErrorScalePxIdentity)))
    print('Trans Pred Median {} Px'.format(np.median(ErrorTransPxPred)))
    print('Trans Identity Median {} Px'.format(np.median(ErrorTransPxIdentity)))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(ErrorScalePxIdentity, 'b.', ErrorScalePxPred, 'r.')
    ax2.plot(ErrorTransPxIdentity, 'b.', ErrorTransPxPred, 'r.')
    plt.show()

if __name__=="__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ReadPath', dest='ReadPath', default='/home/nitin/Datasets/MSCOCO/Test1/PredOuts.txt',\
                         help='Path to load Predictions from, Default:/home/nitin/Datasets/MSCOCO/Test1/PredOuts.txt')

                        
    Args = Parser.parse_args()
    ReadPath = Args.ReadPath
    
    main(ReadPath)
