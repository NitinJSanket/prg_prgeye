# prg_prgeye
Tri-camera module with IMU and dual/triple Cortex M7

## Code Structure

```
Software
└── DeepLearning 
    ├── HomographyNetUnsup
    |   └── *
    ├── RealData
    |   └── *
    ├── SimilarityNet
    |   └── *
    ├── SuperPoint
    |   └── *
    └── SpeedTests
        └── *
```

- `HomographyNetUnsup` has the depracated code from EVDodgeNet.
- `RealData` has the code for evaluation on data from simulation and real world.
- `SimilarityNet` has the code for training and evaluation for all the scenarios presented in the paper.
- `SpeedTests` has the code for testing speed of various networks.

### `SimilarityNet`
To train use `TrainSimilarityNet.py`. The following command line flags can be used.
- `--BasePath`: Base path from where images are loaded, eg., /home/nitin/Datasets/MSCOCO/train2014Processed.
- `--NumEpochs`: Number of epochs the training will be done for
- `--DivTrain`: Factor to reduce Train data by per epoch, used for debugging only or for super large datasets.
- `--MiniBatchSize`: Size of the MiniBatch to use.
- `--LoadCheckPoint`: Load Model from latest Checkpoint from CheckPointPath?
- `--RemoveLogs`: Delete log Files from ./Logs? (BUGGY, DON'T USE).
- `--LossFuncName`: Choice of Loss functions, choose from SL2 (Supervised L2 loss), PhotoL1 (Photometric L1), PhotoChab (Photometric Chabonnier), SSIM (SSIM Loss from GeoNet), SSIMTF (TF's Implementation of GeoNet's SSIM loss). Any loss can be suffixed with HP or SP to compute loss on Highpassed image or SuperPoint cornerness respectively.
- `--RegFuncName`: Choice of regularization function, choose from None, SP (Cornerness from SuperPoint) or HP (Highpassed image). Total loss is computed as LossFunc + sum(Alpha_i * RegFunc_i)
- `--NetworkType`: Choice of Network type, choose from Small, Large.
- `--NetworkName`: Name of network file, eg., Network.VanillaNet. Change the line `VN = Net.xxx` in  `TrainOperation` function to use the correct class name of the network.
- `--CheckPointPath`: Path to save checkpoints.
- `--LogsPath`: Path to save Logs.
- `--GPUDevice`: What GPU do you want to use? -1 for CPU.
- `--DataAug`: Depracated, DO NOT USE. Instead Augment data before and create a dataset.
- `--LR`: Learning Rate.
- `--InitNeurons`: Number of Neurons in the first layer
- `--Input`: Input, choose from I: RGB Images, G: Grayscale Images, HP: HP Grayscale Images, SP: Cornerness from SuperPoint.
    
## References for Images
- [Benchmark Analysis of Representative Deep Neural Network Architectures](https://arxiv.org/abs/1810.00736)
- [An Analysis Of Deep Neural Network Models For Practical Applications](https://arxiv.org/abs/1605.07678)

