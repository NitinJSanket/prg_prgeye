#include <Wire.h>
#include <TimerOne.h>
#include <VL53L1X.h>

VL53L1X sensor;
int val, val2=0;
// Initial time
long int ti;
volatile bool intFlag=false;

#define    MPU9250_ADDRESS            0x68
//#define    MAG_ADDRESS                0x0C

//#define    GYRO_FULL_SCALE_250_DPS    0x00  
//#define    GYRO_FULL_SCALE_500_DPS    0x08
#define    GYRO_FULL_SCALE_1000_DPS   0x10
//#define    GYRO_FULL_SCALE_2000_DPS   0x18

//#define    ACC_FULL_SCALE_2_G        0x00  
#define    ACC_FULL_SCALE_4_G        0x08
//#define    ACC_FULL_SCALE_8_G        0x10
//#define    ACC_FULL_SCALE_16_G       0x18



// This function read Nbytes bytes from I2C device at address Address. 
// Put read bytes starting at register Register in the Data array. 
void I2Cread(uint8_t Address, uint8_t Register, uint8_t Nbytes, uint8_t* Data)
{
  // Set register address
  Wire.beginTransmission(Address);
  Wire.write(Register);
  Wire.endTransmission();
  
  // Read Nbytes
  Wire.requestFrom(Address, Nbytes); 
  uint8_t index=0;
  while (Wire.available())
    Data[index++]=Wire.read();
}


// Write a byte (Data) in device (Address) at register (Register)
void I2CwriteByte(uint8_t Address, uint8_t Register, uint8_t Data)
{
  // Set register address
  Wire.beginTransmission(Address);
  Wire.write(Register);
  Wire.write(Data);
  Wire.endTransmission();
}





// Initializations
void setup()
{
  // Arduino initializations
  Serial.begin(230400);
  Wire.begin();
  Wire.setClock(400000);
  sensor.setTimeout(50);
  if (!sensor.init())
  {
    Serial.println("Failed to detect and initialize sensor!");
    while (1);
  }
  
  // Use long distance mode and allow up to 50000 us (50 ms) for a measurement.
  // You can change these settings to adjust the performance of the sensor, but
  // the minimum timing budget is 20 ms for short distance mode and 33 ms for
  // medium and long distance modes. See the VL53L1X datasheet for more
  // information on range and timing limits.
  sensor.setDistanceMode(VL53L1X::Long);
  sensor.setMeasurementTimingBudget(13000);

  // Start continuous readings at a rate of one measurement every 50 ms (the
  // inter-measurement period). This period should be at least as long as the
  // timing budget.
  sensor.startContinuous(50);

  
  //////////////////////////////////////////////////////////////////////////// 

  
//  Serial.println("IMU Starting...");
  // Set accelerometers low pass filter at 5Hz
  I2CwriteByte(MPU9250_ADDRESS,29,0x06);
  // Set gyroscope low pass filter at 5Hz
  I2CwriteByte(MPU9250_ADDRESS,26,0x06);
 
  
  // Configure gyroscope range
  I2CwriteByte(MPU9250_ADDRESS,27,GYRO_FULL_SCALE_1000_DPS);
  // Configure accelerometers range
  I2CwriteByte(MPU9250_ADDRESS,28,ACC_FULL_SCALE_4_G);
  // Set by pass mode for the magnetometers
  I2CwriteByte(MPU9250_ADDRESS,0x37,0x02);
  
   pinMode(13, OUTPUT);
  Timer1.initialize(10000);         // initialize timer1, and set a 1/2 second period
  Timer1.attachInterrupt(callback);  // attaches callback() as a timer overflow interrupt
  
  
  // Store initial time
  ti=millis();
  
  //Starup delay
  delay(100);    
}


// Counter
long int cpt=0;

void callback()
{ 
  intFlag=true;
  digitalWrite(13, digitalRead(13) ^ 1);
}

// Main loop, read and display data
void loop()
{
  Serial.print("#");
  val = sensor.read();
  if(val!=0)
  {
    Serial.print(val);
    val2=val;
  }
  else
  {
    Serial.print(val2);
  }
  Serial.print (",");
 
  // ____________________________________
  // :::  accelerometer and gyroscope ::: 

  // Read accelerometer and gyroscope
  uint8_t Buf[14];
  I2Cread(MPU9250_ADDRESS,0x3B,14,Buf);
  
  // Create 16 bits values from 8 bits data
  
  // Accelerometer
  int16_t ax=-(Buf[0]<<8 | Buf[1]);
  int16_t ay=-(Buf[2]<<8 | Buf[3]);
  int16_t az=Buf[4]<<8 | Buf[5];

  // Gyroscope
  int16_t gx=-(Buf[8]<<8 | Buf[9]);
  int16_t gy=-(Buf[10]<<8 | Buf[11]);
  int16_t gz=Buf[12]<<8 | Buf[13];
  
    // Display values
  
  // Accelerometer
//  Serial.print("aX: ");
  Serial.print (ax,DEC); 
  Serial.print (","); 
//  Serial.print("aY: ");
  Serial.print (ay,DEC);
  Serial.print (",");
//  Serial.print("aZ: ");
  Serial.print (az,DEC);
  Serial.print (",");  
//  // Gyroscope

//  Serial.print("gX: ");
  Serial.print (gx,DEC);
  Serial.print (","); 
//  Serial.print ("\t");
//  Serial.print("gY: ");
  Serial.print (gy,DEC);
  Serial.print (",");
//  Serial.print ("\t");
//  Serial.print("gZ: ");
  Serial.print (gz,DEC);
  Serial.print (",");  
//  Serial.print ("\t");

  // End of line
  Serial.print("\n");
}
