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


## References for Images
- [Benchmark Analysis of Representative Deep Neural Network Architectures](https://arxiv.org/abs/1810.00736)
- [An Analysis Of Deep Neural Network Models For Practical Applications](https://arxiv.org/abs/1605.07678)

