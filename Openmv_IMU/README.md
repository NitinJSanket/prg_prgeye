# Micropython test code for I2C devices 
The I2C devide used is Sparkfun IMU Breakout [MPU-9250](https://www.sparkfun.com/products/13762)).

## Caveats about OpenMv cam 
- If any file/data needs to be saved on the OpenMv cam module save it in the SD card and not the internal memory of the module
- When the OpenMv cam module is plugged into the system, file explorer should automatically open. If this doesn't happen then there's something wrong and you should refer to [OpenMv documentation](http://docs.openmv.io/index.html). 

## Pin connections
The pin connections can be clearly seen in the images below.

<!-- ![Imu connections](https://github.com/NitinJSanket/prg_prgeye/blob/master/Images/IMU_conn.jpg)
![Openmv side 1 connections](https://github.com/NitinJSanket/prg_prgeye/blob/master/Images/openmv_i2c_conn_1.jpg)
![Openmv side 2 connections](https://github.com/NitinJSanket/prg_prgeye/blob/master/Images/openmv_i2c_conn_2.jpg) -->

- <b> IMU Connections </b>:
<img src="https://github.com/NitinJSanket/prg_prgeye/blob/master/Images/IMU_conn.jpg" width="500">

- <b> Openmv side 1 connections </b>:
<img src="https://github.com/NitinJSanket/prg_prgeye/blob/master/Images/openmv_i2c_conn_1.jpg" width="500">

- <b> Openmv side 2 connections </b>:
<img src="https://github.com/NitinJSanket/prg_prgeye/blob/master/Images/openmv_i2c_conn_2.jpg" width="500">
