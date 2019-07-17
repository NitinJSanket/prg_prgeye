#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from time import sleep
import serial


def serial_pub():
	pub = rospy.Publisher('chatter', String, queue_size=10)
	rospy.init_node('talker', anonymous=True)
	# rate = rospy.Rate(10) # 10hz

	# Establish the connection on a specific port
	ser = serial.Serial('/dev/ttyACM0', 230400)

	while not rospy.is_shutdown():
	    arduino_str = ser.readline() # Read the newest output from the Arduino
	    # rospy.loginfo(arduino_str)
	    pub.publish(arduino_str)
	    # rate.sleep()

if __name__ == '__main__':
	try:
		serial_pub()
	except rospy.ROSInterruptException:
		pass
