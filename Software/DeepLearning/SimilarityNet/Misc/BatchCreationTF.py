import random
import os
import cv2
import numpy as np
import tensorflow as tf
import Misc.ImageUtils as iu
import Misc.warpICSTN2 as warp2
import Misc.MiscUtils as mu


class BatchGeneration():
    def __init__(self, sess, WarpI1PatchIdealGen, IOrgPH, HPH):
        self.sess = sess
        self.WarpI1PatchIdealGen = WarpI1PatchIdealGen
        self.IOrgPH = IOrgPH
        self.HPH = HPH
        
    def RandSimilarityPerturbationTF(self, I1, HObj, PatchSize, MiniBatchSize, ImageSize=None, Vis=False):
        if(ImageSize is None):
            ImageSize = np.array(np.shape(I1))[1:]
            # TODO: Extract MiniBatchSize here

        H, Params = HObj.GetRandReducedHICSTN()

        # Maybe there is a better way? https://dominikschmidt.xyz/tensorflow-data-pipeline/
        
        FeedDict = {self.IOrgPH: I1, self.HPH: H}
        I2 = np.uint8(self.sess.run([self.WarpI1PatchIdealGen], feed_dict=FeedDict)[0]) # self.WarpI1PatchIdealGen.eval(feed_dict=FeedDict)

        # Crop in center for PatchSize
        P1 = iu.CenterCrop(I1, PatchSize)
        P2 = iu.CenterCrop(I2, PatchSize)

        if(Vis is True):
            for count in range(MiniBatchSize):
                A = IOrgBatch[count]
                B = WarpI1IdealRet[count]
                AP = P1[count]
                BP = P2[count]
                cv2.imshow('I1, I2', np.hstack((A, B)))
                cv2.imshow('P1, P2', np.hstack((AP, BP)))
                cv2.waitKey(0)

        # P1 is I1 cropped to patch Size
        # P2 is I1 Crop Warped (I2 Crop)
        # H is Homography
        # Params is the stuff H is made from 
        return I1, I2, P1, P2, H, Params


    def GenerateBatchTF(self, TrainNames, PatchSize, MiniBatchSize, HObj, BasePath, OriginalImageSize):
        """
        Inputs: 
        DirNames - Full path to all image files without extension
        NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
        TrainLabels - Labels corresponding to Train
        NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
        ImageSize - Size of the Image
        MiniBatchSize is the size of the MiniBatch
        Outputs:
        I1Batch - Batch of I1 images after standardization and cropping/resizing to ImageSize
        HomeVecBatch - Batch of Homing Vector labels
        """
        
        IOrgBatch = [] 

        
        ImageNum = 0
        while ImageNum < MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(TrainNames)-1)
            RandImageName = BasePath + os.sep + TrainNames[RandIdx] 
            I = cv2.imread(RandImageName)
            I = iu.RandomCrop(I, OriginalImageSize)
            if (I is None):
                continue
            ImageNum += 1
            IOrgBatch.append(I)

        # Cast Lists as np arrays
        IOrgBatch = np.array(IOrgBatch)
        # Similarity and Patch generation 
        I1Batch, I2Batch, P1Batch, P2Batch, HBatch, ParamsBatch = self.RandSimilarityPerturbationTF(IOrgBatch, HObj, PatchSize, MiniBatchSize, ImageSize = None, Vis = False)
            
        ICombined = np.concatenate((P1Batch[:,:,:,0:3], P2Batch[:,:,:,0:3]), axis=3)
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IBatch = iu.StandardizeInputs(np.float32(ICombined))

        return IBatch, I1Batch, I2Batch, P1Batch, P2Batch, HBatch, ParamsBatch

