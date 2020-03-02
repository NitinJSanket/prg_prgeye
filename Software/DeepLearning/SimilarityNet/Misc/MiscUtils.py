import time
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import importlib

# Don't generate pyc codes
sys.dont_write_bytecode = True

def tic():
    StartTime = time.time()
    return StartTime

def toc(StartTime):
    return time.time() - StartTime

def remap(x, oMin, oMax, iMin = None, iMax = None):
    #range check
    if oMin == oMax:
        print("Warning: Zero output range")
        return None

    if iMin is None:
          iMin = np.amin(x)

    if iMax is None:
          iMax = np.amax(x)

    if iMin == iMax:
        print("Warning: Zero input range")
        return None

    result = np.add(np.divide(np.multiply(x - iMin, oMax - oMin), iMax - iMin), oMin)

    return result

def FindLatestModel(CheckPointPath):
    FileList = glob.glob(CheckPointPath + '*.ckpt.index') # * means all if need specific format then *.csv
    LatestFile = max(FileList, key=os.path.getctime)
    # Strip everything else except needed information
    LatestFile = LatestFile.replace(CheckPointPath, '')
    LatestFile = LatestFile.replace('.ckpt.index', '')
    return LatestFile

def convertToOneHot(vector, n_labels):
    return np.equal.outer(vector, np.arange(n_labels)).astype(np.float)

def RotMatError(R1, R2):
    # https://math.stackexchange.com/questions/2581668/error-measure-between-two-rotations-when-one-matrix-might-not-be-a-valid-rotatio?noredirect=1&lq=1
    return np.linalg.norm(np.matmul(np.matrix.transpose(R1), R2) - np.eye(3), ord='fro')

def isRotMat(R):
    # Original Code from: https://stackoverflow.com/questions/53808503/how-to-test-if-a-matrix-is-a-rotation-matrix
    # square matrix test
    # Bug Fix by: Nitin J. Sanket
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(np.matmul(R, R.transpose()), np.identity(R.shape[0], np.float), rtol=1e-03, atol=1e-03)
    should_be_one = np.allclose(np.linalg.det(R), 1.0, rtol=1e-03, atol=1e-03)
    return should_be_identity and should_be_one

def TransError(T1, T2):
    return np.linalg.norm(T1-T2)

def ClosestRotMat(RDash):
    U, S, Vt = np.linalg.svd(RDash, full_matrices=False)
    SModified = np.eye(3)
    SModified[2,2] = np.linalg.det(np.matmul(U, Vt))
    Rot = np.matmul(np.matmul(U, Vt), Vt)
    return Rot

# Python program to print 
# colored text and background
# Adapted from: https://www.geeksforgeeks.org/print-colors-python-terminal/
def printcolor(s, color = 'r'):
    if(color == 'r'):
         print("\033[91m {}\033[00m" .format(s)) # red
    elif(color == 'g'):
         print("\033[92m {}\033[00m" .format(s)) # green
    elif(color == 'y'):
        print("\033[93m {}\033[00m" .format(s)) # yellow
    elif(color == 'lp'):
         print("\033[94m {}\033[00m" .format(s)) # light purple
    elif(color == 'p'):
        print("\033[95m {}\033[00m" .format(s)) # purple
    elif(color == 'c'):
         print("\033[96m {}\033[00m" .format(s)) # cyan
    elif(color == 'lgray'):
        print("\033[97m {}\033[00m" .format(s)) # light gray
    elif(color == 'k'):
        print("\033[98m {}\033[00m" .format(s)) # black


# Doesnt work
# def ImportModule(s):
#     # get a handle on the module
#     mdl = importlib.import_module(s)

#     # is there an __all__?  if so respect it
#     if "__all__" in mdl.__dict__:
#         names = mdl.__dict__["__all__"]
#     else:
#         # otherwise we import all names that don't begin with _
#         names = [x for x in mdl.__dict__ if not x.startswith("_")]

#     # now drag them in
#     globals().update({k: getattr(mdl, k) for k in names})
