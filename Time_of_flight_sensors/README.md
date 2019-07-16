# Time of flight sensors- VL53L1X and VL53L0X
## Setup
### WARNING!!
The code has been tested for **Arduino uno** and **Arduino MEGA 2560** using Arduino IDE version **1.8.9**

### ISSUES!
* The code doesn't compile with old Arduino IDE. It was compiled on Arduino 1.0.6 and compilation failed.
* Further the code doesn't work on Arduino micro.

### Hardware

The hardware connections are fairly simple. Only the relevant pin connections are shown in the description below:
```
Arduino        VL53L1X board
-------        -------------
2.6V to 5.5V - VIN
         GND - GND
         SDA - SDA
         SCL - SCL
```

Verbose desciption of all pins is provided below:

| PIN   | Description                                                                                                                                                                                                                           |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VDD   | Regulated 2.8 V output. Almost 150 mA is available to power external components. (If you want to bypass the internal regulator, you can instead use this pin as an input for voltages between 2.6 V and 3.5 V with VIN disconnected.) |
| VIN   | This is the main 2.6 V to 5.5 V power supply connection. The SCL and SDA level shifters pull the I²C lines high to this level.                                                                                                        |
| GND   | The ground (0 V) connection for your power supply. Your I²C control source must also share a common ground with this board.                                                                                                           |
| SDA   | Level-shifted I²C data line: HIGH is VIN, LOW is 0 V                                                                                                                                                                                  |
| SCL   | Level-shifted I²C clock line: HIGH is VIN, LOW is 0 V                                                                                                                                                                                 |
| XSHUT | This pin is an active-low shutdown input; the board pulls it up to VDD to enable the sensor by default. Driving this pin low puts the sensor into hardware standby. This input is not level-shifted.                                  |
| GPIO1 | Programmable interrupt output (VDD logic level). This output is not level-shifted.                                                                                                                                                    |


### Software
#### Libraries
* Use the libraries provided in the **Libraries** directory
* Unzip the files if you have downloaded the library from the official repo and copy it to the "library" folder of arduino.[Warning: There can be multiple "library" folders for arduino. Make sure to install the library in right directory.]
* Restart arduino IDE (if it was previously running)

#### CODE
* Launch the Arduino IDE and open the code that you need to run (i.e VL53L0X OR VL53L1X). 
* Go to tools in the Arduino IDE menu and select appropriate port and the name of Arduino board you are using.
* Upload the code to Arduino(using ctrl+u) and open serial monitor (using ctrl+shift+m).
