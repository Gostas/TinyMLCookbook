import numpy as np
import serial
import cv2

ser = serial.Serial()
ser.port = '/dev/ttyACM0'
ser.baudrate = 115200

ser.open()
ser.reset_input_buffer()

cam = cv2.VideoCapture(0)

def serial_readline(obj):
    data = obj.readline()
    return data.decode("utf-8").strip()

while True:
    data_str = serial_readline(ser)
    if str(data_str) == '<cam-read>':
        _, img_bgr = cam.read()
        h0,w0 = img_bgr.shape[:2]
        h1 = min(h0, w0)
        w1 = h1
        img_cropped = img_bgr[0:w1, 0:h1]
        img_resized = cv2.resize(img_cropped, (76,76), interpolation=cv2.INTER_LINEAR)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        data = bytearray(img_gray.astype(np.uint8))
        ser.write(data)
        cv2.imshow('Captured image', img_resized)
        data_str = serial_readline(ser)
        num_objs = int(data_str)
        for x in range(num_objs):
            data_str = serial_readline(ser)
            xy_str = data_str.split(',')
            xy = [int(xy_str[0], xy_str[1])
            cv2.circle(img_resized, xy, 4, (0,0,255), -1)
    key = cv2.waitKey(33)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cam.release()
