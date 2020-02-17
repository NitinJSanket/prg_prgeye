# prg_prgeye
Tri-camera module with IMU and dual/triple Cortex M7

## TODO
- Log Num Neurons as well
- Test MobileNet
- Test ShuffleNetv2
- HP Filter in TF
- Cornerness warp in TF

## Checklist
- [x] Num ICSTN Blocks and which warping function to use (On Vanilla Network)
  - [x]  2 psudo-similarity 
  - [x]  2 scale 2 translation (half size each)
  - [x]  2 translation 2 scale (half size each) 
  - [x]  4 psudo-similarity (half size each) 
- [ ] Architectures (Choose best Num ICSTN setup)
  - [x]  Vanilla Net Arch
  - [ ]  SqueezeNet 
  - [ ]  ResNet 
  - [ ]  MobileNet
  - [ ]  ShuffleNetv2
- [ ] Loss Functions (Choose best ICSTN combination on best Arch)
   - [ ]  Supervised ICSTN 
   - [ ]  Unsupervised L1 
   - [ ]  Unsupervised L1 with Cornerness
   - [ ]  Unsupervised L1 with HP
   - [ ]  Unsupervised L1 with HP + Cornerness
   - [ ]  Unsupervised Chab
   - [ ]  Unsupervised Barron
   - [ ]  Unsupervised Barron with best of Cornerness, HP, HP + Cornerness
   - [ ]  Supervised Events DB + Chab
- [ ] Compression (Bigger Network <= 25 MB, Smaller Network <= 2.5 MB)
  - [ ]  Student Teacher on best Arch + ICSTN Num + Loss Func
  - [ ]  Weight Pruning (tflite) on best Arch + ICSTN Num + Loss Func
  - [ ]  Quantization (tflite) on best Arch + ICSTN Num + Loss Func
  - [ ]  Model Distillation: Student Teacher with Projection Loss
  - [ ]  Direct dropping number of weights
- [ ] Speed Tests
  - [ ] Desktop PC (CPU + GPU) on bigger and smaller network (most accurate from above), Batch Size of 1
      - [ ]  TF (CPU) Smaller Network
      - [ ]  TF (CPU) Bigger Network
      - [ ]  TF-Lite (CPU) Smaller Network
      - [ ]  TF-Lite (CPU) Bigger Network
      - [ ]  TF (GPU) Smaller Network
      - [ ]  TF (GPU) Bigger Network
      - [ ]  TF-Lite (GPU) Smaller Network
      - [ ]  TF-Lite (GPU) Bigger Network
  - [ ] NanoPi Neo Core 2 on bigger and smaller network (most accurate from above), Batch Size of 1
      - [ ]  TF (CPU) Smaller Network
      - [ ]  TF (CPU) Bigger Network
      - [ ]  TF-Lite (CPU) Smaller Network
      - [ ]  TF-Lite (CPU) Bigger Network
      - [ ]  Coral USB Stick Smaller Network
      - [ ]  Coral USB Stick Bigger Network
  - [ ]  Sipeed Maix on smaller network (most accurate from above), Batch Size of 1
  - [ ]  Google Coral Dev Board bigger and smaller network (most accurate from above), Batch Size of 1
    - [ ]  TF-Lite Smaller Network
    - [ ]  TF-Lite Bigger Network
  - [ ] Intel Up Board on bigger and smaller network (most accurate from above), Batch Size of 1
	- [ ]  TF (CPU) Smaller Network
    - [ ]  TF-Lite (CPU) Smaller Network
    - [ ]  TF (CPU) Bigger Network
    - [ ]  TF-Lite (CPU) Bigger Network
  - [ ] NVIDIA Jetson TX2 on bigger and smaller network (most accurate from above), Batch Size of 1
    - [ ]  TF (CPU) Smaller Network
    - [ ]  TF (CPU) Bigger Network
    - [ ]  TF-Lite (CPU) Smaller Network
    - [ ]  TF-Lite (CPU) Bigger Network
    - [ ]  TF (GPU) Smaller Network
    - [ ]  TF (GPU) Bigger Network
    - [ ]  TF-Lite (GPU) Smaller Network
    - [ ]  TF-Lite (GPU) Bigger Network
  - [ ]  Odometry Data for 5 sequences using Vicon
  - [ ]  KF for fusion for odom
  - [ ]  Grayscale vs RGB (Minor difference)
  - [ ] Comparisons With normal images, error induced due to augmentation
    - [ ]  Correlation Flow
    - [ ]  Chatterjee Flow
    - [ ]  SIFT
    - [ ]  ORB
    - [ ]  KLT 

- How are Model Sizes Computed? (From Num. params or file size)?

Reference papers:
- [Benchmark Analysis of Representative Deep Neural Network Architectures](https://arxiv.org/abs/1810.00736)
- [An Analysis Of Deep Neural Network Models For Practical Applications](https://arxiv.org/abs/1605.07678)

