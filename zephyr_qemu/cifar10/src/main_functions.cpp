#include <cstdint>
#include "main_functions.h"
#include "input.h"
#include "model.h"

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

/* Globals, used for compatibility with Arduino-style sketches. */
namespace {
	const tflite::Model *model = nullptr;
	tflite::MicroInterpreter *interpreter = nullptr;
	TfLiteTensor *input = nullptr;
	TfLiteTensor *output = nullptr;

	constexpr int tensor_arena_size = 44000;
	uint8_t tensor_arena[tensor_arena_size];
}  /* namespace */

float o_scale = 0.0f;
int32_t o_zero_point = 0;

/* The name of this function is important for Arduino compatibility. */
void setup(void)
{
	/* Map the model into a usable data structure. This doesn't involve any
	 * copying or parsing, it's a very lightweight operation.
	 */
	model = tflite::GetModel(cifar10_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		MicroPrintf("Model provided is schema version %d not equal "
					"to supported version %d.",
					model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}

	/* This pulls in the operation implementations we need.
	 * NOLINTNEXTLINE(runtime-global-variables)
	 */
	static tflite::MicroMutableOpResolver <6> resolver;
	resolver.AddFullyConnected();
	resolver.AddConv2D();
	resolver.AddDepthwiseConv2D();
	resolver.AddRelu();
	resolver.AddMaxPool2D();
	resolver.AddReshape();

	/* Build an interpreter to run the model with. */
	static tflite::MicroInterpreter static_interpreter(
		model, resolver, tensor_arena, tensor_arena_size);
	interpreter = &static_interpreter;

	/* Allocate memory from the tensor_arena for the model's tensors. */
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		MicroPrintf("AllocateTensors() failed");
		return;
	}

	/* Obtain pointers to the model's input and output tensors. */
	input = interpreter->input(0);
	output = interpreter->output(0);

	const auto* o_quantization = reinterpret_cast<TfLiteAffineQuantization*>
		(output->quantization.params);
	o_scale = o_quantization->scale->data[0];
	o_zero_point = o_quantization->zero_point->data[0];
}

/* The name of this function is important for Arduino compatibility. */
void loop(void)
{
	for(int32_t i = 0; i < g_test_len; ++i) {
		input->data.int8[i] = g_test[i];
	}
	interpreter->Invoke();
	int32_t idx_max = 0;
	float pb_max = 0;

	for(int32_t idx = 0; idx < 10; ++idx) {
		int8_t out_val = output->data.int8[idx];
		float pb = (float)(out_val - o_zero_point) * o_scale;
		if(pb > pb_max){
			pb_max = pb;
			idx_max = idx;
		}
	}
	if(idx_max == g_test_ilabel) {
		MicroPrintf("PASSED!\n");
	}
	else {
		MicroPrintf("FAILED! Classification = %d\n", idx_max);
	}
	while(1);
}
