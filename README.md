# prg_prgeye
Tri-camera module with IMU and dual/triple Cortex M7

## TODO
- [ ] Architectures
  - [ ] Vanilla Net Arch
  - [ ] SqueezeNet or SqueezeNext
  - [ ] ResNet or ResNeXt
  - [ ] DenseNet
- [ ] Num ICSTN Blocks and which warping function to use (Choose best Arch)
  - [ ] 2 psudo-similarity
  - [ ] 1 scale 1 translation 1 scale 1 translation (half size each) 
  - [ ] 4 psudo-similarity (half size each)
- [ ] Loss Functions (Choose best ICSTN combination on best Arch)
   - [ ] Supervised ICSTN 
   - [ ] Unsupervised L1 
   - [ ] Unsupervised L1 with Cornerness
   - [ ] Unsupervised L1 with HP
   - [ ] Unsupervised L1 with HP + Cornerness
   - [ ] Unsupervised Chab
   - [ ] Unsupervised Barron
   - [ ] Unsupervised Barron with best of Cornerness, HP, HP + Cornerness
   - [ ] Supervised Events DB + Chab
- [ ] Compression (Bigger Network <= 25 MB, Smaller Network <= 2.5 MB)
  - [ ] Student Teacher on best Arch + ICSTN Num + Loss Func
  - [ ] Weight Pruning (tflite) on best Arch + ICSTN Num + Loss Func
  - [ ] Quantization (tflite) on best Arch + ICSTN Num + Loss Func
  - [ ] Model Distillation: Student Teacher with Projection Loss
  - [ ] Direct dropping number of weights
- [ ] Speed Tests
  - [ ] Desktop PC (CPU + GPU) on bigger and smaller network (most accurate from above), Batch Size of 1
      - [ ] TF (CPU) Smaller Network
      - [ ] TF (CPU) Bigger Network
      - [ ] TF-Lite (CPU) Smaller Network
      - [ ] TF-Lite (CPU) Bigger Network
      - [ ] TF (GPU) Smaller Network
      - [ ] TF (GPU) Bigger Network
      - [ ] TF-Lite (GPU) Smaller Network
      - [ ] TF-Lite (GPU) Bigger Network
  - [ ] NanoPi Neo Core 2 on bigger and smaller network (most accurate from above), Batch Size of 1
      - [ ] TF (CPU) Smaller Network
      - [ ] TF (CPU) Bigger Network
      - [ ] TF-Lite (CPU) Smaller Network
      - [ ] TF-Lite (CPU) Bigger Network
      - [ ] NCS v1/v2 Smaller Network
      - [ ] NCS v1/v2 Bigger Network
  - [ ] Sipeed Maix on smaller network (most accurate from above), Batch Size of 1
  - [ ] Google Coral Dev Board bigger and smaller network (most accurate from above), Batch Size of 1
    - [ ] TF-Lite Smaller Network
    - [ ] TF-Lite Bigger Network
  - [ ] Intel Up Board on bigger and smaller network (most accurate from above), Batch Size of 1
    - [ ] TF (CPU) Smaller Network
    - [ ] TF (CPU) Bigger Network
    - [ ] TF-Lite (CPU) Smaller Network
    - [ ] TF-Lite (CPU) Bigger Network
  - [ ] NVIDIA Jetson TX2 on bigger and smaller network (most accurate from above), Batch Size of 1
    - [ ] TF (CPU) Smaller Network
    - [ ] TF (CPU) Bigger Network
    - [ ] TF-Lite (CPU) Smaller Network
    - [ ] TF-Lite (CPU) Bigger Network
    - [ ] TF (GPU) Smaller Network
    - [ ] TF (GPU) Bigger Network
    - [ ] TF-Lite (GPU) Smaller Network
    - [ ] TF-Lite (GPU) Bigger Network
  - [ ] Odometry Data for 5 sequences using Vicon
  - [ ] KF for fusion for odom
  - [ ] Grayscale vs RGB (Minor difference)
  - [ ] Comparisons With normal images, error induced due to augmentation
    - [ ] Correlation Flow
    - [ ] Chatterjee Flow
    - [ ] SIFT
    - [ ] ORB
    - [ ] klt


- How are Model Sizes Computed? (From Num. params or file size)?


## Hardware

### H6 board
- dimension limit 35x35mm
- remove all peripherals 
- Keep CSI ports, USB ports, serial ports
- headers all around or ribbon cables like jevois

### Components
#### Camera Modules 
| Module | Sourcing Link | Price | Datasheet | Module | Price($) |Additional Details |
| --- | --- | --- | --- |  --- | --- | --- |
| ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) MT9V034 | [Digikey](https://www.digikey.com/catalog/es/partgroup/mt9v034/75147)  | 21.25 | [Link](http://www.onsemi.com/pub/Collateral/MT9V034-D.PDF) | [Link](http://www.uctronics.com/arducam-cmos-mt9v034-1-3-inch-0-36mp-monochrome-camera-module.html)| 49.99 | 752x480 at 60fps, Global Shutter, Used in [OpenMV Cam](https://openmv.io/products/openmv-cam-m7), Probably used in [MyntEye](https://mynteyeai.com/products/mynt-eye-stereo-camera) |
| NOIP1SN1300A (Python 1300) | [Newark](https://www.newark.com/on-semiconductor/noip1sn1300a-qdi/image-sensor-monochrome-lcc-48/dp/02AC3796?CMP=AFC-OP) | 103.23 | [Link](http://www.onsemi.com/pub/Collateral/NOIP1SN1300A-D.PDF) | | | 1280x1024 at 43fps, Global Shutter, Used in [Open Visual Computer](https://arxiv.org/pdf/1809.07674.pdf) |
#### MT9V034 Camera driver for linux
https://github.com/torvalds/linux/blob/master/drivers/media/i2c/mt9v032.c

#### Inertial Measurement Modules (IMUs)
| Module | Sourcing Link | Price | Datasheet | Additional Details | More info | Breakout Board | Price($) | Comments|
| --- | --- | --- | --- |  --- | --- |--- | --- | --- |
| BMI088 | [Digikey](https://www.digikey.com/product-detail/en/bosch-sensortec/BMI088/828-1082-1-ND/8634942) | 7.99 | [Link](https://ae-bst.resource.bosch.com/media/_tech/media/product_flyer/Bosch_Sensortec_Product_flyer_BMI088.pdf) | 3 Axis Acc + Gyro | | |
| ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) ICM-20789 | [Digikey](https://www.digikey.com/en/product-highlight/i/invensense/icm-20789-pressure-sensor) | 9.09 | [Link](http://www.invensense.com/wp-content/uploads/2017/10/DS-000169-ICM-20789-TYP-v1.3.pdf) | 3 Axis Acc + Gyro + 1 Axis Pressure | Recommended by Invensense for Drones, Newer |[Link](https://www.notwired.co/ProductDetail/NWMOTICM20789-notWired-co/605573/)| 18.95 | |
| MPU-9250 | [Digikey](https://www.digikey.com/product-detail/en/tdk-invensense/MPU-9250/1428-1019-1-ND/4626450) | 10.6| [Link](https://store.invensense.com/datasheets/invensense/MPU9250REV1.0.pdf) | 3 Axis Acc + Gyro + Compass|Recommended by Invensense for Drones, Common| |
| VMU931 | [Variense](https://variense.com/product/vmu931/) | 99.0 | [Link](http://variense.com/Docs/VMU931/specification_sheet_VMU931.pdf) | 3 Axis Acc + Gyro + Compass | Angles estimated directly, Self bias compensation | | |[word from manufacturer] Without aluminuim housing the dimension is 28mm in diameter and the cost is $85(+$15 shipping), so no go |


#### ARM Microcontrollers/Microprocessors
| Module | Sourcing Link | Price | Datasheet | Additional Details | Developtment Breakout board | DevBoard Price |
| --- | --- | --- | --- |  --- | --- | --- |
| STM32H753VIH6 | [Mouser](https://www.mouser.com/ProductDetail/STMicroelectronics/STM32H753VIH6?qs=%2fha2pyFaduh09l6hj91PU9oGd521L6LfDAtowUQ9At1xFSxsRRKEUA%3d%3d) | 14 | [Link](https://www.st.com/resource/en/datasheet/stm32h753vi.pdf) | Also available in different package as [STM32H753VIT6](https://www.mouser.com/ProductDetail/STMicroelectronics/STM32H753VIT6?qs=%2fha2pyFaduh09l6hj91PU4SsTEbwmG%252b1PE8cmadUoz5rVxtTE6ezFQ%3d%3d) | [STM32H753I-EVAL](https://www.mouser.com/ProductDetail/STMicroelectronics/STM32H753I-EVAL?qs=sGAEpiMZZMtw0nEwywcFgJjuZv55GFNmfu2%2fjBD%2f4fawdHCCf0qIzg%3d%3d)|$ 462.02 |
| ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) STM32H743VIH6  | [Mouser](https://www.mouser.com/ProductDetail/STMicroelectronics/STM32H743VIH6?qs=%2fha2pyFadujiWVHRlW6sBVJFKnRr%252bVzOL9BR8UM%252brrpLnn1Hy6YNUg%3d%3d) | 13.62 | [Link](https://www.mouser.com/datasheet/2/389/stm32h743bi-1156566.pdf) | Used on [OpenMV Cam H7](https://openmv.io/products/openmv-cam-h7), Also available in different package as [STM32H743VIT6](https://www.mouser.com/ProductDetail/STMicroelectronics/STM32H743VIT6?qs=%2fha2pyFadujiWVHRlW6sBS19o1KOsEoNOPHYwQMB6s6uRcfwYi8MwQ%3d%3d) |1][STM32H743I-EVAL](https://www.mouser.com/ProductDetail/STMicroelectronics/STM32H743I-EVAL?qs=sGAEpiMZZMtw0nEwywcFgJjuZv55GFNmxBOleZvxxoEfyTJIVNPTJQ%3d%3d) 2][STM32H743ZI](https://www.mouser.com/ProductDetail/STMicroelectronics/NUCLEO-H743ZI?qs=5aG0NVq1C4zVqdFc0FeE%252bw%3d%3d) |$460, $23 |
| NXP i.MX RT1050 | [Digikey](https://www.digikey.com/product-detail/en/nxp-usa-inc/MIMXRT1052DVL6A/568-13515-ND/7646297) | 5.6 | [Link](https://www.nxp.com/docs/en/data-sheet/IMXRT1050CEC.pdf) | 4.5x faster on CMSIS using [MicroPython port](https://github.com/RockySong/micropython-rocky/tree/omv_initial_integrate) | [MIMXRT1050-EVK: i.MX RT1050 Evaluation Kit](https://www.nxp.com/support/developer-resources/run-time-software/i.mx-developer-resources/i.mx-rt1050-evaluation-kit:MIMXRT1050-EVK)| US$79.00 |
|  |  |  |  |  | [IMX RT1052 DEVELOPER'S KIT](http://www.embeddedartists.com/products/kits/imxrt1052_kit.php) | €149  |

#### STM32H743VIH6 Specs
| Functionality | # | PIN No. |
| --- | --- | --- |
| SPI| 4 | SPI_1_MOSI(88), SPI_1_MISO(90), SPI_2_MOSI(16), SPI_2_MISO(53), SPI_3_MOSI(36), SPI_3_MISO(79), SPI_4_MOSI(44), SPI_4_MISO(04) |

#### STM32H743ZIH6 Eval kitSpecs
| Functionality | # | PIN No. |
| --- | --- | --- |
| SPI| 5 | --- |
#### Distance Sensor
| Module | Sourcing Link | Price | Datasheet | Breakout Board | Price($) | Additional Details |
| --- | --- | --- | --- |  --- | --- | --- |
|  ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) VL53L0X | [Mouser](https://www2.mouser.com/ProductDetail/STMicroelectronics/VL53L0CXV0DH-1?qs=dTJS0cRn7ojtsK3C9%252bTaSw==) | 4.52 | [Link](https://www.st.com/resource/en/datasheet/vl53l0x.pdf) | 1][Link](https://www.robotshop.com/en/tof-range-finder-sensor-breakout-board-voltage-regulator-vl53l0x.html?gclid=EAIaIQobChMIvau_2KD83QIVgUOGCh3Hbg32EAQYBCABEgJunvD_BwE) 2][Link](https://solarbotics.com/product/51111/)| 1] 9.95 2]13.95 |Very Short Range, Used on Crazyflie(range: min[3cm] - max[2m]) |
|  ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) VL53L1X | [Digikey](https://www.digikey.com/product-detail/en/stmicroelectronics/VL53L1CXV0FY-1/497-17764-1-ND/8276742?cur=USD&lang=en) | 6.62 | [Link](https://www.st.com/content/ccc/resource/technical/document/datasheet/group3/7d/85/c8/95/fb/3b/4e/2d/DM00452094/files/DM00452094.pdf/jcr:content/translations/en.DM00452094.pdf) | [Link](https://www.sparkfun.com/products/14722) | 19.95 |Single Beam, 27 degree FOV, 0.8m-4.0m range |
| OPT3101 | [DigiKey](https://www.digikey.com/product-detail/en/texas-instruments/OPT3101RHFT/296-50203-1-ND/9487227) [Mouser Evaluation Kit](https://www2.mouser.com/ProductDetail/Texas-Instruments/OPT3101EVM?qs=%252bEew9%252b0nqrCCoKpiJlizOg%3D%3D) | 11.1/170.97 | [Link](http://www.ti.com/lit/ds/symlink/opt3101.pdf) | | |0.3-5.0m range, 2% error, 4KHz | 
| OPT8241 | [Mouser](https://www2.mouser.com/ProductDetail/Texas-Instruments/OPT8241NBN?qs=sGAEpiMZZMvdy8WAlGWLcKPa4nk0O3C%252bwIQ%2fiSDKtJw%3d) | 50.21 | [Link](http://www.ti.com/lit/ds/symlink/opt8241.pdf) | | |320 x 240 1/3" grid TOF sensor at 150Hz, 1m-5m range | 
| OPT8320 | [Mouser](https://www2.mouser.com/ProductDetail/Texas-Instruments/OPT8320NBP?qs=sGAEpiMZZMvt1VFuCspEMqRfqAj1Q9FPWWzOxVfju8g%3d) | 42.30 | [Link](http://www.ti.com/lit/ds/symlink/opt8320.pdf) | | |80 x 60 1/6" grid TOF sensor at 1000Hz | 

#### Other Computers running Linux at a similar compute budget
- [Omega Onion 2](https://onion.io/omega2/)
- [VoCore v2](https://vocore.io/v2.html)


## Resources
- The FSYNC Pin for IMU Hardware sync (Check page 35): [Link](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2405959/15750_FULLTEXT.pdf?sequence=1)


## Software

### Neural Network Based Homography Computation
Code can be found in [Software/DeepLearning/HomographyNetUnsup](Software/DeepLearning/HomographyNetUnsup). Follow these instructions for training and testing.

#### Training
- Download MS-COCO dataset from [here](http://cocodataset.org/#home). We'll use the train2014 set for training.
- Prepare text files for training by running the [DataParse.py](Software/DeepLearning/HomographyNetUnsup/DataPrase.py) code. Specify ``--ReadPath`` and ``--WritePath`` as the path to COCO dataset. For eg. ``/home/nitin/Datasets/MSCOCO/train2014``.
- Run the [TrainHomographyNet.py](Software/DeepLearning/HomographyNetUnsup/TrainHomographyNet.py) and specify the following command line flags. You can use execute ``./TrainHomographyNet.py -h`` to obtain the help given below.
  - ``--BasePath``: Path where Images are stored. For this eg. ``/home/nitin/Datasets/MSCOCO/train2014``.
  - ``--DivTrain``: Factor to reduce Train data by per epoch. Used only for debugging.
  - ``--MiniBatchSize``: Size of the MiniBatch to use.
  - ``--LoadCheckPoint``: Resume Training by loading latest model from CheckPointPath.
  - ``--RemoveLogs``: Depracated. DO NOT USE.
  - ``--LossFuncName``: Choice of Loss functions, choose from PhotoL1, PhotoChab, PhotoRobust. Used only for Unsupervised training.
  - ``--NetworkType``: Choice of Network type, choose from Small, Large. Use Large for training on Images. Small network doesn't have enough capacity to learn anything on images but works on event frames.
  - ``--CheckPointPath``: Path to save checkpoints.
  - ``--LogsPath``: Path to save TensorBoard Logs.
  - ``--GPUDevice``: Use specific GPU Number when you have multiple GPUs. Use -1 for running on CPU.
  - ``--LR``: Learning Rate.
  - ``TrainingType``: S for Supervised and US for unsupervised. Uses Loss function from ``--LossFuncName`` when training in US mode. Uses L2 loss in S mode.
- Recommended to train about 50-100 Epochs at Batch Size of 32-256.

#### Testing
- Not implemented yet!
