import time
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True

def tic():
    StartTime = time.time()
    return StartTime

def toc(StartTime):
    return time.time() - StartTime

def remap(x, oMin, oMax, iMin, iMax):
    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if iMin == iMax:
        print("Warning: Zero output range")
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
