# Arduino and OpenGL libraries for IMU9250 IMU:
- Tested with Arduino Micro and IMU9250 Adafruit breakout board.
`Arduino: 1.8.9 (Windows Store 1.8.21.0) (Windows 10), Board: "Arduino/Genuino Micro"`

[Reference](https://bitbucket.org/cinqlair/mpu9250)

- In Arduino Micro, always initialize `Wire.begin()` after `Serial.begin()`. 
- Runs at about `180 Hz`. Serial Monitor makes it a little slower to `110Hz`.

**NOTE: RANDOM BUG (CAN'T BE FIXED I GUESS)**:<br>
If you encounter an error like this:
```
exit status 1
Error compiling for board Arduino/Genuino Micro.
```
Remove the `Serial.println("...");` statements. Most likely, the problem lies in `"..."` in print functions. 
