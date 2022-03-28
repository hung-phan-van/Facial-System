import socket
import time
# from imutils.video import VideoStream
import imagezmq
import cv2
vid = cv2.VideoCapture(0) 

SERVER_ADDR  = "tcp://172.16.131.79:5003"


sender = imagezmq.ImageSender(connect_to=SERVER_ADDR)

time.sleep(2.0)  # allow camera sensor to warm up
count = 0 
while True:  # send images as stream until Ctrl-C
    _, image = vid.read()
    count += 1
    sender.send_image(str(count), image)
