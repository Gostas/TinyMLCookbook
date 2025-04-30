#include <Arduino_OV767X.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>
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

int32_t bytes_per_frame, bytes_per_pixel;
byte data[160 * 120 * 2]; // QVGA: 320x240 X 2 bytes per pixel (RGB565)
//byte frame[48*48*3];

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

    float res = (v00 * wx1 * wy1);
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
    Serial.begin(115200);
    while (!Serial);

    if (!Camera.begin(QQVGA, RGB565, 1)) {
        Serial.println("Failed to initialize camera!");
        while (1);
    }

    bytes_per_frame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
    bytes_per_pixel = Camera.bytesPerPixel();

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

// original frame is 160x120
// crop to 120x120
// resize to 48x48
// rescale values to range [-1,1]
// quantize
void loop() {
    int32_t idx = 0;
    Camera.readFrame(data);
    
    // perform backward mapping
    for(size_t yo = 0; yo < height_o; ++yo){
        float yi_f = (yo * scale_y);
        int32_t yi = (int32_t)std::floor(yi_f);
        for(size_t xo = 0; xo < width_o; ++xo){
            float xi_f = (xo * scale_x);
            int32_t xi = (int32_t)std::floor(xi_f);
            int32_t x0 = xi, y0 = yi;
            int32_t x1 = std::min(x0 + 1, width_i - 1),
                    y1 = std::min(y0 + 1, height_i - 1);
            int32_t stride_x = bytes_per_pixel,
                    stride_y = Camera.width() * bytes_per_pixel;

            uint8_t rgb00[3], rgb01[3], rgb10[3], rgb11[3];
            
            rgb565_rgb888(&data[stride_x*x0 + stride_y*y0], rgb00);
            rgb565_rgb888(&data[stride_x*x0 + stride_y*y1], rgb01);
            rgb565_rgb888(&data[stride_x*x1 + stride_y*y0], rgb10);
            rgb565_rgb888(&data[stride_x*x1 + stride_y*y1], rgb11);

            uint8_t c_i; float c_f; int8_t c_q;
            for(uint8_t i = 0; i < 3; ++i){
                c_i = bilinear(rgb00[i], rgb01[i],
                            rgb10[i], rgb11[i],
                            xi_f, yi_f);
                //frame[idx] = c_i;
                c_f = rescale((float)c_i, 1.f/255.f, -1.f);
                c_q = quantize(c_f, tflu_scale, tflu_zeropoint);
                tflu_i_tensor->data.int8[idx++] = c_q;
            }
        }
    }
    //MicroPrintf("<image>\n%d\n%d", width_o, height_o);
    //Serial.write(frame, height_o*width_o*3);
    tflu_interpreter->Invoke();
    const char *label[] = {"book", "mug", "unknown"};
    MicroPrintf("\n%s | %s  | %s", label[0], label[1], label[2]);
    Serial.print(tflu_o_tensor->data.f[0]);
    Serial.print(" | ");
    Serial.print(tflu_o_tensor->data.f[1]);
    Serial.print(" | ");
    Serial.println(tflu_o_tensor->data.f[2]);
    //Serial.println("</image>");    
}
