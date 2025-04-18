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

byte data[160 * 120 * 2]; // QVGA: 320x240 X 2 bytes per pixel (RGB565)

template <typename T>
inline T clamp_0_255(T x){
    return std::max(std::min(x, (T)255), (T)0);
}

void rgb565_rgb888(uint8_t *in, uint8_t *out){
    uint16_t p = (in[0] << 8) | in[1];
    out[0] = ((p >> 11) & 0x1f) << 3;
    out[1] = ((p >> 5) & 0x3f) << 2;
    out[2] = (p & 0x1f) << 3;
}

void ycbcr422_rgb888(int32_t Y, int32_t Cb, int32_t Cr, uint8_t *out){
    Cr -= 128;
    Cb -= 128;
    int32_t r, g, b;

    r = Y + Cr + (Cr >> 2) + (Cr >> 3) + (Cr >> 5);
    g = Y - ((Cb >> 2) + (Cb >> 4) + (Cb >> 5)) - ((Cr >> 1) + (Cr >> 3) + (Cr >> 4));
    b = Y + Cb + (Cb >> 1) + (Cb >> 2) + (Cb >> 6);
    out[0] = clamp_0_255(r);
    out[1] = clamp_0_255(g);
    out[2] = clamp_0_255(b);
}

void setup() {
    button.mode(PullUp);
    led = 0;
    Serial.begin(115200);
    while (!Serial);

    if (!Camera.begin(QQVGA, YUV422, 1)) {
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
        uint8_t rgb888[3];
        int32_t step_bytes = Camera.bytesPerPixel() * 2;

        Camera.readFrame(data);
        
        Serial.println("<image>");
        Serial.println(Camera.width());
        Serial.println(Camera.height());

        for(int32_t i = 0; i < bytes_per_frame; i+= step_bytes){
            const int32_t Y0 = data[i];
            const int32_t Cr = data[i+1];
            const int32_t Y1 = data[i+2];
            const int32_t Cb = data[i+3];

            ycbcr422_rgb888(Y0, Cb, Cr, rgb888);

            Serial.println(rgb888[0]);
            Serial.println(rgb888[1]);
            Serial.println(rgb888[2]);

            ycbcr422_rgb888(Y1, Cb, Cr, rgb888);

            Serial.println(rgb888[0]);
            Serial.println(rgb888[1]);
            Serial.println(rgb888[2]);
        }
        //Serial.write(data, bytes_per_frame);
        Serial.println("</image>");
    }
}
