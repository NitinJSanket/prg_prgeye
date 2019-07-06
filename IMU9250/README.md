# Arduino and OpenGL libraries for IMU9250 IMU:
- Tested with Arduino Micro and IMU9250 Adafruit breakout board.

[Reference](https://bitbucket.org/cinqlair/mpu9250)

- In Arduino Micro, always initialize `Wire.begin()` after `Serial.begin()`. 
- Runs at about `180 Hz`. Serial Monitor makes it a little slower to `110Hz`.
