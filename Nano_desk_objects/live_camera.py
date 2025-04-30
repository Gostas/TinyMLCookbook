import numpy as np
import serial
import cv2

ser = serial.Serial()
ser.port = '/dev/ttyACM0'
ser.baudrate = 115200

ser.open()
ser.reset_input_buffer()

def serial_readline(obj):
    data = obj.readline()
    return data.decode("utf-8").strip()

while True:
    data_str = serial_readline(ser)
    if str(data_str) == '<image>':
        w_str = serial_readline(ser)
        h_str = serial_readline(ser)
        w = int(w_str)
        h = int(h_str)
        c = int(3)

        image = np.empty((h, w, c), dtype=np.uint8)
        # read pixel values where R,G,B are 1 byte each
        buffer = ser.read(w*h*3)
        data_str1 = serial_readline(ser)
        data_str2 = serial_readline(ser)
        data_str2 = serial_readline(ser)
        data_str3 = serial_readline(ser)
        print(data_str1)
        print(data_str2)
        if str(data_str3) == '</image>':
            for i in range(0, h*w*c):
                y = int(i/(c*w))
                x = int((i/c) % w)
                d = 2-int(i%c) # convert to BGR format for open cv
                image[y][x][d] = buffer[i]        
            cv2.imshow("capture", image)
            cv2.waitKey(25)
