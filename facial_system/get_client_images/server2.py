# run this program on the Mac to display image streams from multiple RPis
import cv2
import imagezmq
import time
SERVER_ADDR  = os.environ("SERVER_ADDR")
image_hub = imagezmq.ImageHub(SERVER_ADDR)
while True:  # show streamed images until Ctrl-C
    name, image = image_hub.recv_image()
    print(name, image.shape)
    time.sleep(20)
    image_hub.send_reply(b'OK')
    