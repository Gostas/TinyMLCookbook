/*
  OV767X - Camera Capture Raw Bytes

  This sketch reads a frame from the OmniVision OV7670 camera
  and writes the bytes to the Serial port. Use the Procesing
  sketch in the extras folder to visualize the camera output.

  Circuit:
    - Arduino Nano 33 BLE board
    - OV7670 camera module:
      - 3.3 connected to 3.3
      - GND connected GND
      - SIOC connected to A5
      - SIOD connected to A4
      - VSYNC connected to 8
      - HREF connected to A1
      - PCLK connected to A0
      - XCLK connected to 9
      - D7 connected to 4
      - D6 connected to 6
      - D5 connected to 5
      - D4 connected to 3
      - D3 connected to 2
      - D2 connected to 0 / RX
      - D1 connected to 1 / TX
      - D0 connected to 10

  This example code is in the public domain.
*/

#include <Arduino_OV767X.h>
#include "mbed.h"

int bytes_per_frame;

mbed::DigitalIn button(p30);
mbed::DigitalOut led(p13);
constexpr int PRESSED = 0;

byte data[320 * 240 * 2]; // QVGA: 320x240 X 2 bytes per pixel (RGB565)

void rgb565_rgb888(uint8_t *in, uint8_t *out){
    uint16_t p = (in[0] << 8) | in[1];
    out[0] = ((p >> 11) & 0x1f) << 3;
    out[1] = ((p >> 5) & 0x3f) << 2;
    out[2] = (p & 0x1f) << 3;
}

void setup() {
    button.mode(PullUp);
    led = 0;
    Serial.begin(115200);
    while (!Serial);

    if (!Camera.begin(QVGA, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
    }

    bytes_per_frame = Camera.width() * Camera.height() * Camera.bytesPerPixel();

    // Optionally, enable the test pattern for testing
    Camera.testPattern();
}

void loop() {
    led = !button;
    if(button == PRESSED){
        Camera.readFrame(data);
        uint8_t rgb888[3];
        Serial.println("<image>");
        Serial.println(Camera.width());
        Serial.println(Camera.height());
        // int32_t bytes_per_pixel = Camera.bytesPerPixel();
        // int32_t i = 0;
        // for(; i < bytes_per_frame; i+= bytes_per_pixel){
        //     rgb565_rgb888(&data[i], rgb888);
        //     Serial.println(rgb888[0]);
        //     Serial.println(rgb888[1]);
        //     Serial.println(rgb888[2]);
        // }
        Serial.write(data, bytes_per_frame);
        Serial.println("</image>");
    }
}
