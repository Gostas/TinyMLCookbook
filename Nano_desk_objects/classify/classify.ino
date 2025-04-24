#include <Arduino_OV767X.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "mbed.h"
#include "model.h"

const tflite::Model *tflu_model = nullptr;
tflite::MicroInterpreter *tflu_interpreter = nullptr;
TfLiteTensor *tflu_i_tensor = nullptr, *tflu_o_tensor = nullptr;

constexpr int tensor_arena_size = 128000;
alignas(16) uint8_t tensor_arena[tensor_arena_size];

float tflu_scale = 0.0f;
int32_t tflu_zeropoint = 0;

namespace {
    using ClassifierOpResolver = tflite::MicroMutableOpResolver<8>;

    TfLiteStatus RegisterOps(ClassifierOpResolver& op_resolver) {
        TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
        TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
        TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
        TF_LITE_ENSURE_STATUS(op_resolver.AddRelu6());
        TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
        TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
        TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
        TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
        return kTfLiteOk;
    }
}  // namespace

int32_t height_i = 120, width_i = 120;
int32_t height_o = 48, width_o = 48;

float scale_x = (float)width_i / (float)width_o;
float scale_y = scale_x;

int bytes_per_frame;

mbed::DigitalOut led(p13);

const int32_t bytes_per_pixel = Camera.bytesPerPixel();
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

uint8_t bilinear(uint8_t v00, uint8_t v01, uint8_t v10,
                uint8_t v11, float xi_f, float yi_f){
    const float xi = (int32_t)std::floor(xi_f);
    const float yi = (int32_t)std::floor(yi_f);

    const float wx0 = (xi_f - xi);
    const float wx1 = (1.f - wx0);

    const float wy0 = (yi_f - yi);
    const float wy1 = (1.f - wy0);

    float res = 0;
    res += (v00 * wx1 * wy1);
    res += (v10 * wx0 * wy1);
    res += (v01 * wx1 * wy0);
    res += (v11 * wx0 * wy0);

    return clamp_0_255(res);
}

float rescale(float x, float scale, float offset){
    return (x * scale) - offset;
}

int8_t quantize(float x, float scale, float zero_point) {
    return (x / scale) + zero_point;
}

void setup() {
    led = 0;
    Serial.begin(115200);
    while (!Serial);

    if (!Camera.begin(QQVGA, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
    }

    bytes_per_frame = Camera.width() * Camera.height() * Camera.bytesPerPixel();

    tflu_model = tflite::GetModel(model_tflite);
    if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            tflu_model->version(), TFLITE_SCHEMA_VERSION);
        return;
      }

    // register relevant ops
    static ClassifierOpResolver tflu_op_resolver;
    TfLiteStatus s = RegisterOps(tflu_op_resolver);
    if(s != kTfLiteOk){
        MicroPrintf("op_resolver error");
        return;
    }

    static tflite::MicroInterpreter static_interpeter(tflu_model, 
                                                    tflu_op_resolver,
                                                    tensor_arena,
                                                    tensor_arena_size);
    tflu_interpreter = &static_interpeter;
    tflu_interpreter->AllocateTensors();
    tflu_i_tensor = tflu_interpreter->input(0);
    tflu_o_tensor = tflu_interpreter->output(0);

    const auto *i_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);

    tflu_scale = i_quant->scale->data[0];
    tflu_zeropoint = i_quant->zero_point->data[0];
}

void loop() {
    //crop
    //rescale
    int32_t idx = 0;
    for(int32_t yo = 0; yo < height_o; ++yo){
        float yi_f = (yo * scale_y);
        int32_t yi = (int32_t)std::floor(yi_f);
        for(int xo = 0; xo < width_o; ++xo){
            float xi_f = (xo * scale_x);
            int32_t xi = (int32_t)std::floor(xi_f);
        }
    }
    //quantize

    Camera.readFrame(data);
    
    Serial.println("<image>");
    Serial.println(Camera.width());
    Serial.println(Camera.height());


    Serial.write(data, bytes_per_frame);
    Serial.println("</image>");
}
