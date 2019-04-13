# Micropython test code for I2C devices
The code written is test the functionality of I2C protocol with a third party device.
The I2C devide used is Sparkfun IMU Breakout [MPU-9250](https://www.sparkfun.com/products/13762). Another resource for the device is [register map](https://cdn.sparkfun.com/assets/learn_tutorials/5/5/0/MPU-9250-Register-Map.pdf), which can be used to find  address for the device and it's components.

## Caveats about OpenMv cam 
- If any file/data needs to be saved on the OpenMv cam module save it in the SD card and not the internal memory of the module
- When the OpenMv cam module is plugged into the system, file explorer should automatically open. If this doesn't happen then there's something wrong and you should refer to [OpenMv documentation](http://docs.openmv.io/index.html). 

## Pin connections
The pin connections can be clearly seen in the images below.
OpenMV cam                IMU
     |                     |
    P4 ------------------ SCL
    P5 ------------------ SCA
    3.3v ---------------- VDD
    GND ----------------- GND
    

- <b> IMU Connections </b>:
<img src="https://github.com/NitinJSanket/prg_prgeye/blob/master/Images/IMU_conn.jpg" width="500">

- <b> Openmv side 1 connections </b>:
<img src="https://github.com/NitinJSanket/prg_prgeye/blob/master/Images/openmv_i2c_conn_1.jpg" width="500">

- <b> Openmv side 2 connections </b>:
<img src="https://github.com/NitinJSanket/prg_prgeye/blob/master/Images/openmv_i2c_conn_2.jpg" width="500">

## Instruction to run the code
- Clone the repository in your system
- Download the OpenmvIDE using the [link](https://openmv.io/pages/download) and follow the installation instructions.
- Run OpenmvIDE and open the **i2c_IMU.py** inside the folder *Openmv_IMU* 
- Assuming that the OpenMv module is connected, click the *connect* button on the bottom left corner of the IDE
- Once the module is connected, press the run botton right below the *connect* button.
