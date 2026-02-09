# This program is meant to be run on remote devices (other laptops) that will connect their video streams to the main laptop

import cv2
import socket
import imagezmq

# Enter the IP address of the host laptop
# The host can find their IP by running `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
host_ip = ""

# connect_to specifies the IP address of your main laptop (the host)
sender = imagezmq.ImageSender(connect_to='tcp://{host_ip}}:5555') 

# Give this device a unique name
rpi_name = socket.gethostname()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Resize to reduce network lag (Optional but recommended)
    # Sending 1080p video over WiFi will be slow. 640px is usually enough for face detection
    frame = cv2.resize(frame, (640, 480))
    
    # Send the frame
    # The host will receive (rpi_name, frame)
    sender.send_image(rpi_name, frame)