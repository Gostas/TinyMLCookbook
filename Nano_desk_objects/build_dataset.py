import numpy as np
import serial
from PIL import Image
import uuid
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

ser = serial.Serial()
ser.port = '/dev/ttyACM0'
ser.baudrate = 115200

label = ''
gdrive_id = ''

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
        # for i in range(0, h*w*c):
        #     y = int(i/(c*w))
        #     x = int((i/c) % w)
        #     d = int(i%c)
        #     data_str = serial_readline(ser)
        #     image[y][x][d] = int(data_str)

        buffer = ser.read(w*h*2)
        # convert RGB565 to RGB888
        for i in range(0, 2*h*w, 2):
            p = (buffer[i] << 8) | buffer[i+1]
            r = int(((p >> 11) & 0x1f) << 3)
            g = int(((p >> 5) & 0x3f) << 2)
            b = int((p & 0x1f) << 3)
            y = int(i/(2*w))
            x = int((i/2) % w)
            image[y][x][:3] = r,g,b
        data_str = serial_readline(ser)
        if str(data_str) == '</image>':
            image_pil = Image.fromarray(image)
            crop_area = (0, 0, h, h)
            image_cropped = image_pil.crop(crop_area)
            image_cropped.show()
            key = input('Save image? [y] for YES: ')
            if key == 'y':
                str_label = """Provide the label's name or leave it empty to use [{}]:""".format(label)
                label_new = input(str_label)
                if label_new != '':
                    label = label_new
                unique_id = str(uuid.uuid4())
                filename = label + '_' + unique_id + '.png'
                image_cropped.save(filename)
                str_gid = """Provide the Google Drive folder ID or leave it empty to use [{}]""".format(gdrive_id)
                gdrive_id_new = input(str_gid)
                if gdrive_id != '':
                    gdrive_id = gdrive_id_new
                gfile = drive.CreateFile({'parents': [{'id': gdrive_id}]})
                gfile.SetContentFile(filename)
                gfile.Upload()