# prg_prgeye
Tri-camera module with IMU and dual/triple Cortex M7

## Components
#### Camera Modules 
| Module | Sourcing Link | Price | Datasheet | Module | Price($) |Additional Details |
| --- | --- | --- | --- |  --- | --- | --- |
| ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) MT9V034 | [Digikey](https://www.digikey.com/catalog/es/partgroup/mt9v034/75147)  | 21.25 | [Link](http://www.onsemi.com/pub/Collateral/MT9V034-D.PDF) | [Link](http://www.uctronics.com/arducam-cmos-mt9v034-1-3-inch-0-36mp-monochrome-camera-module.html)| 49.99 | 752x480 at 60fps, Global Shutter, Used in [OpenMV Cam](https://openmv.io/products/openmv-cam-m7), Probably used in [MyntEye](https://mynteyeai.com/products/mynt-eye-stereo-camera) |
| NOIP1SN1300A (Python 1300) | [Newark](https://www.newark.com/on-semiconductor/noip1sn1300a-qdi/image-sensor-monochrome-lcc-48/dp/02AC3796?CMP=AFC-OP) | 103.23 | [Link](http://www.onsemi.com/pub/Collateral/NOIP1SN1300A-D.PDF) | | | 1280x1024 at 43fps, Global Shutter, Used in [Open Visual Computer](https://arxiv.org/pdf/1809.07674.pdf) |

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


#### Distance Sensor
| Module | Sourcing Link | Price | Datasheet | Breakout Board | Price($) | Additional Details |
| --- | --- | --- | --- |  --- | --- | --- |
|  ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) VL53L0X | [Mouser](https://www2.mouser.com/ProductDetail/STMicroelectronics/VL53L0CXV0DH-1?qs=dTJS0cRn7ojtsK3C9%252bTaSw==) | 4.52 | [Link](https://www.st.com/resource/en/datasheet/vl53l0x.pdf) | 1][Link](https://www.robotshop.com/en/tof-range-finder-sensor-breakout-board-voltage-regulator-vl53l0x.html?gclid=EAIaIQobChMIvau_2KD83QIVgUOGCh3Hbg32EAQYBCABEgJunvD_BwE) 2][Link](https://solarbotics.com/product/51111/)| 1] 9.95 2]13.95 |Very Short Range, Used on Crazyflie |
|  ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) VL53L1X | [Digikey](https://www.digikey.com/product-detail/en/stmicroelectronics/VL53L1CXV0FY-1/497-17764-1-ND/8276742?cur=USD&lang=en) | 6.62 | [Link](https://www.st.com/content/ccc/resource/technical/document/datasheet/group3/7d/85/c8/95/fb/3b/4e/2d/DM00452094/files/DM00452094.pdf/jcr:content/translations/en.DM00452094.pdf) | [Link](https://www.sparkfun.com/products/14722) | 19.95 |Single Beam, 27 degree FOV, 0.8m-4.0m range |
| OPT3101 | [DigiKey](https://www.digikey.com/product-detail/en/texas-instruments/OPT3101RHFT/296-50203-1-ND/9487227) [Mouser Evaluation Kit](https://www2.mouser.com/ProductDetail/Texas-Instruments/OPT3101EVM?qs=%252bEew9%252b0nqrCCoKpiJlizOg%3D%3D) | 11.1/170.97 | [Link](http://www.ti.com/lit/ds/symlink/opt3101.pdf) | | |0.3-5.0m range, 2% error, 4KHz | 
| OPT8241 | [Mouser](https://www2.mouser.com/ProductDetail/Texas-Instruments/OPT8241NBN?qs=sGAEpiMZZMvdy8WAlGWLcKPa4nk0O3C%252bwIQ%2fiSDKtJw%3d) | 50.21 | [Link](http://www.ti.com/lit/ds/symlink/opt8241.pdf) | | |320 x 240 1/3" grid TOF sensor at 150Hz, 1m-5m range | 
| OPT8320 | [Mouser](https://www2.mouser.com/ProductDetail/Texas-Instruments/OPT8320NBP?qs=sGAEpiMZZMvt1VFuCspEMqRfqAj1Q9FPWWzOxVfju8g%3d) | 42.30 | [Link](http://www.ti.com/lit/ds/symlink/opt8320.pdf) | | |80 x 60 1/6" grid TOF sensor at 1000Hz | 


## Resources
- The FSYNC Pin for IMU Hardware sync (Check page 35): [Link](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2405959/15750_FULLTEXT.pdf?sequence=1)
