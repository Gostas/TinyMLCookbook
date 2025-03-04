#define FRAME_SIZE (76*76)
uint8_t frame[FRAME_SIZE];

void setup(){
    Serial.begin(115200);
    while(!Serial);
}

void loop(){
    Serial.println("<cam-read>");
    Serial.readBytes(frame, FRAME_SIZE);
}