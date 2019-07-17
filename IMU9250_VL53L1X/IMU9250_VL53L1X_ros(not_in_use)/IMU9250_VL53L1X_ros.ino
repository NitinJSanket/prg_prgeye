#include <Wire.h>
#include <TimerOne.h>
#include <VL53L1X.h>
#include <ros.h>
#include <stdio.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
// #include <std_msgs/Float64.h>
#include <geometry_msgs/Twist.h>

VL53L1X sensor;
// int my_time = 0;
// int IMU_Dist_Freq_Ratio = 20;


int val, val2=0;

ros::NodeHandle  nh;
// std_msgs::String str_msg;
// Variable to store data from all sensros
std_msgs::Float32 dist;
std_msgs::Float32 imu_data;

// sensor_msgs::Imu all_sensor_msg;
// std_msgs::Float32 temp_msg;
ros::Publisher chatter("imu_lidar_data/dist", &dist);
ros::Publisher imu("imu_lidar_data/imu", &imu_data);



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



// Initial time
long int ti;
volatile bool intFlag=false;

// Initializations
void setup()
{
  // Arduino initializations
  nh.initNode();
  nh.advertise(chatter);
  nh.advertise(imu);
  
  
//  Serial.begin(115200);
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
  sensor.setMeasurementTimingBudget(50000);

  // Start continuous readings at a rate of one measurement every 50 ms (the
  // inter-measurement period). This period should be at least as long as the
  // timing budget.
  sensor.startContinuous(50);

  
  //////////////////////////////////////////////////////////////////////////// 

  
//  Serial.println("IMU Starting...");
  nh.loginfo("IMU Starting...");
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
//  delay(1000);
  
  // Store initial time
  ti=millis();


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
  
  // std_msgs::Float64 cov[];

  //read data from LIDAR (VL53L1X)
  val = sensor.read();
  if(val!=0)
  {
//    Serial.print(val);/
    val2=val;
    dist.data = val;
  }
  else
  {
//    Serial.print(val2);/
    dist.data = val2;
  }
  
  // all_sensor_msg.orientation.x = 0;
  // all_sensor_msg.orientation.y = 0;
  // all_sensor_msg.orientation.z = 0;
  
  
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

  imu_data.data = ax;

  
//  imu_data.linear.x = ax;
//  imu_data.linear.y = ay;
//  imu_data.linear.z = az;
//  
//  imu_data.angular.x = gx;
//  imu_data.angular.y = gy;
//  imu_data.angular.z = gz;
  // all_sensor_msg.linear_acceleration.x = ax;
  // all_sensor_msg.linear_acceleration.y = ay;
  // all_sensor_msg.linear_acceleration.z = az;

  // all_sensor_msg.angular_velocity.x = gx;
  // all_sensor_msg.angular_velocity.y = gy;
  // all_sensor_msg.angular_velocity.z = gz;

    // Display values
//  char val[6];
  // Accelerometer
//  Serial.print("aX: ");
//  strcat(string_to_pub,"aX: ");
//  Serial.print (ax,DEC); 
//  dtostrf(ax, 5, 1, val);
//  strcat(string_to_pub,val);
//  Serial.print ("\t");
//  strcat(string_to_pub,"\t");
  
//  Serial.print("aY: ");
//  strcat(string_to_pub,"aY: ");
//  Serial.print (ay,DEC);
//  dtostrf(ay, 5, 1, val);
//  strcat(string_to_pub,val);
//  Serial.print ("\t");
//  strcat(string_to_pub,"\t");
  
//  Serial.print("aZ: ");
//  strcat(string_to_pub,"aZ: ");
//  Serial.print (az,DEC);  
//  dtostrf(az, 5, 1, val);
//  strcat(string_to_pub,val);
//  Serial.print ("\t");
//  strcat(string_to_pub,"\t");
////  Serial.print ("\t");
//  strcat(string_to_pub,"\t");
////  Serial.print ("\t");
//  strcat(string_to_pub,"\t");

//String acc = "aX: = "+String(ax)+"\t"+"ay: = "+String(ay)+"\t"+"aZ: = "+String(az)+"\t";

//  // Gyroscope

////  Serial.print("gX: ");
//  strcat(string_to_pub,"gX: ");
////  Serial.print (gx,DEC);
//  dtostrf(gx, 5, 1, val);
//  strcat(string_to_pub,val);
////  Serial.print ("\t");
//  strcat(string_to_pub,"\t");
//  
////  Serial.print("gY: ");
//  strcat(string_to_pub,"gY: ");
////  Serial.print (gy,DEC);
//  dtostrf(gy, 5, 1, val);
//  strcat(string_to_pub,val);
////  Serial.print ("\t");
//  strcat(string_to_pub,"\t");
//  
////  Serial.print("gZ: ");
//  strcat(string_to_pub,"gZ: ");
////  Serial.print (gz,DEC);
//  dtostrf(gz, 5, 1, val);
//  strcat(string_to_pub,val);
////  Serial.print ("\t");
//  strcat(string_to_pub,"\t");
//String gyro = "gX: = "+String(gx)+"\t"+"gy: = "+String(gy)+"\t"+"gz: = "+String(gz)+"\t";

  
  // my_time++;
//  Serial.print("Time-Stamp: ");
//  strcat(string_to_pub,"Time-Stamp: ");
//  Serial.print(my_time);
//  dtostrf(my_time, 5, 1, val);
//  strcat(string_to_pub,val);
  // End of line
//  Serial.println("");
//  strcat(string_to_pub,"\n");
//  delay(100);    

//String imu_data = "dist = "+String(dist)+" Dist update: "+String(op)+"\t"+acc+gyro+"Time-Stamp: "+String(my_time)+"\n";
//int length_ = imu_data.indexOf("\n")+2;
//char data_final[length_+1];
//imu_data.toCharArray(data_final,length_+1);

  // temp_msg.data = dist;
//  str_msg.data = data_final;
  chatter.publish( &dist );
  imu.publish(&imu_data);
  nh.spinOnce();
  
}
