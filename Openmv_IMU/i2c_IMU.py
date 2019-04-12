# I2C Control
#
# This example shows how to use the i2c bus on your OpenMV Cam by dumping the
# contents on a standard EEPROM. To run this example either connect the
# Thermopile Shield to your OpenMV Cam or an I2C EEPROM to your OpenMV Cam.

from pyb import I2C
import pyb


i2c = I2C(2, I2C.MASTER) # The i2c bus must always be 2.
#mem = i2c.mem_read(256, 0x50, 0) # The eeprom slave address is 0x50.
# device address
addr = 0x68


# Memory address for each axis of sensor
ACCEL_XOUT0 = const(0x3B)
ACCEL_XOUT_L = const(0x3C)

ACCEL_YOUT0 = const(0x3D)
ACCEL_YOUT_L = const(0x3E)

ACCEL_ZOUT0 = const(0x3F)
ACCEL_ZOUT_L = const(0x40)

GYRO_XOUT0 = const(0x43)
GYRO_YOUT0 = const(0x45)
GYRO_ZOUT0 = const(0x47)


#init buffers
adata = bytearray(6)
gdata = bytearray(6)

# helper for converting 2 bytes to int.
btoi = lambda msb, lsb: (msb << 8 | lsb) if not msb & 0x80 else -(((msb ^ 255) << 8) | (lsb ^ 255) + 1)

# some initial params
accel_range = 16.0
accel_rate = 2048.0
gyro_range = 2000.0
gyro_rate = 16.4
accel = [0.0] * 3
gyro = [0.0] * 3

# open file
#f = open('imu_data.txt','w')

# Number of samples to be observed to calibrate
num_samples = 200
i=0

# variables for calibration
acc_x_sum=0
acc_y_sum=0
acc_z_sum=0

gyro_x_sum=0
gyro_y_sum=0
gyro_z_sum=0

# Start while loop
while True:
    #i2c.recv
    i2c.mem_read(adata, addr, ACCEL_XOUT0)
    i2c.mem_read(gdata, addr, GYRO_XOUT0)
    accel = [btoi(adata[0], adata[1])/accel_rate, btoi(adata[2], adata[3])/accel_rate, btoi(adata[4], adata[5])/accel_rate]
    gyro = [btoi(gdata[0], gdata[1])/gyro_rate, btoi(gdata[2], gdata[3])/gyro_rate, btoi(gdata[4], gdata[5])/gyro_rate]
    #f.write("accel \n")
    if i<=num_samples:
        acc_x_sum+=accel[0]
        acc_y_sum+=accel[1]
        acc_z_sum+=accel[2]

        gyro_x_sum+=gyro[0]
        gyro_y_sum+=gyro[0]
        gyro_z_sum+=gyro[0]

        if i== num_samples:
            acc_x_avg = acc_x_sum/num_samples
            acc_y_avg = acc_y_sum/num_samples
            acc_z_avg = acc_z_sum/num_samples

            gyro_x_avg = gyro_x_sum/num_samples
            gyro_y_avg = gyro_y_sum/num_samples
            gyro_z_avg = gyro_z_sum/num_samples

        print('\n\n')
        print('i= ',i)
        print('accel: ', accel)
        print('gyro: ', gyro)
        print('\n\n')
        pyb.delay(50)
    else:
        accel = [(accel[0]- acc_x_avg)*180/8, (accel[1]- acc_y_avg)*180/8, (accel[2]- acc_z_avg -8.0)*180/8]
        gyro =[gyro[0]-gyro_x_avg, gyro[1]-gyro_y_avg, gyro[2]-gyro_z_avg]
        print('\n\n')
        print('i= ',i)
        print('accel: ', accel)
        print('gyro: ', gyro)
        print('\n\n')
        pyb.delay(50)

    i+=1

#f.close()






#print("\n[")
#for i in range(16):
    #print("\t[", end='')
    #for j in range(16):
        #print("%03d" % mem[(i*16)+j], end='')
        #if j != 15: print(", ", end='')
    #print("]," if i != 15 else "]")
#print("]")
