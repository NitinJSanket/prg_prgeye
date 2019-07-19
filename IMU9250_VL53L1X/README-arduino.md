# Arduino code of IMU9250 and VL53L1X together.
This folder contains the code to get data from two sensors (IMU9250 and VL53L1X) at the same time.

## Installing Modified libraries for VL53L1x
* Go to the [time of flight](https://github.com/NitinJSanket/prg_prgeye/tree/master/Time_of_flight_sensors) directory in the prg_prgeye repo.
* Under the Libraries folder find VL53L1x driver.
* Copy and paste the folder in the libraries folder of your Arduino directory. For help on installing libraries manuall go to the [link](https://learn.adafruit.com/adafruit-all-about-arduino-libraries-install-use/how-to-install-a-library)

## Flashing code on Arduino
* Open the code in Ardunino IDE and connect the arduino using usb. For IMU or LIDAR pin connections checkout [this readme](https://github.com/NitinJSanket/prg_prgeye/blob/master/Time_of_flight_sensors/README.md)
* Select the proper Arduino board and port under the tools menu
* Upload the code using the gui or by using the shortcut ``` ctrl + u ```.


## Reading serial message using ROS
Open a terminal using ``` ctrl + alt + t ```. Type the following command to start roscore 
```
roscore
```
### Launching rosnode using python
* Open a new terminal and navigate to the directory where ```arduino-serial.py``` is located.
* To run type
```
python arduino-serial.py
```

### Launching rosnode using ROS
* Create a catkin workspace using the following commands:
```
mkdir -p ~/catkin_prg_ws/src
cd ~/catkin_prg_ws/
catkin_make
```
* Copy the folder (package) ```arduino_serial_read``` and paste it into the ```src``` folder of the workspace (catkin_prg_ws)
* use catkin_make
```
cd ~/catkin_prg_ws/
catkin_make
```
* source the workspace
```
source ~/catkin_prg_ws/devel/setup.bash
```
* run the node
```
rosrun arduino_serial_read arduino-serial.py
```
* To check the received messages open a new terminal and execute 
```
rostopic echo /imu_lidar_data
```
