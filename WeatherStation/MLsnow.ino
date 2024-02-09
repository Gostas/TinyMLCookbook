#include <Arduino_HTS221.h>
#include <TensorFlowLite.h>
#include <ArduinoBLE.h>
#include "mbed.h"
#include "mbed_events.h"
#include "Thread.h"
#include "Mutex.h"
#include "Watchdog.h"
// the model
#include "q_aware_model.h"
// the interpreter
#include "tensorflow/lite/micro/micro_interpreter.h"
// for MicroPrintf
#include "tensorflow/lite/micro/micro_log.h"
// for DebugLog (used by MicroPrintf)
#include "tensorflow/lite/micro/system_setup.h"
// Register operations used by the model
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// to load the model
#include "tensorflow/lite/schema/schema_generated.h"

//#include "cmsis_os2.h"
//#include "ConditionVariable.h"
//#include "Kernel.h"

// for s, ms, etc.
using namespace std::chrono_literals;

/* for BLE UUIDs
 https://www.bluetooth.com/specifications/assigned-numbers/
 Celsius temperature (degree Celsius) = 0x272F
 percentage = 0x27AD */

// Bluetooth® Low Energy weather service
BLEService weatherService("181A");

// Bluetooth® Low Energy Characteristic
BLEIntCharacteristic tempChar("2A6E",  // standard 16-bit characteristic UUID
    BLERead | BLENotify); // remote clients will be able to get notifications if this characteristic changes
BLEIntCharacteristic humChar("2A6F", BLERead | BLENotify);
BLEIntCharacteristic snowProbChar("2A78", BLERead | BLENotify); // (Displayed as Rainfall)

unsigned long long previousMillis;  // last time data was transmitted

constexpr int NUM_HOURS = 3;  // period of sensing and prediction
constexpr int num_reads = 3;  // number of times to repeat sensor reading
int8_t t_vals[NUM_HOURS], h_vals[NUM_HOURS]; // quantized values
// store current readings and readings previously transmitted
float cur_t, cur_h, pred_f, 
  old_t=-273, old_h=-1, old_pred=-1;

int cur_idx;

// Quantization parameters
float tflu_i_scale = 1.0f;
float tflu_o_scale = 1.0f;
int32_t tflu_i_zero_point = 0;
int32_t tflu_o_zero_point = 0;

// Normalization parameeters for temperature and humidity
constexpr float t_mean = 4.18374f;
constexpr float h_mean = 77.88801f;
constexpr float t_std = 9.80673f;
constexpr float h_std = 14.65571f;

const tflite::Model *tflu_model = nullptr;
tflite::MicroInterpreter *tflu_interpreter = nullptr;
TfLiteTensor *tflu_i_tensor = nullptr;
TfLiteTensor *tflu_o_tensor = nullptr;

// Define how much memory the interpreter will have available
constexpr int tensor_arena_size = 4*1024;
byte tensor_arena[tensor_arena_size]__attribute__((aligned(16)));

// Pull in only the operations needed by the model
namespace {
using MlSnowOpResolver = tflite::MicroMutableOpResolver<3>;

TfLiteStatus RegisterOps(MlSnowOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
  return kTfLiteOk;
}
}  // namespace

void readPredictSend();

inline void scale_and_quantize(float &t, float &h){
  // scale with Z-score
  t = (t - t_mean)/t_std;
  h = (h - h_mean)/h_std;

  // quantize
  t = (t / tflu_i_scale) + (float)tflu_i_zero_point;
  h = (h / tflu_i_scale) + (float)tflu_i_zero_point;
}

events::EventQueue queue;
rtos::Thread eventThread;
rtos::Mutex mutex;
//rtos::ConditionVariable cond(mutex);

void setup() {
  /*
  TODO
    3. Investigate freeze when no connection
    5. Add pressure measurements
    6. Add units to characteristics
    7. consider case when previousMillis > LL max and resets to 0
    8. Create notes
  */
  tflite::InitializeTarget();   // Initializes serial communication

  // init sensor
  if(!HTS.begin()){
    Serial.println("Failed initializations of HTS221!");
    while(1);
  }

  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("WeatherStation");

  // BLEDescriptor celsiusUnit("272F", "°C");
  // BLEDescriptor percentUnit("27AD", "%");

  // tempChar.addDescriptor(celsiusUnit);
  // humChar.addDescriptor(percentUnit);
  // snowProbChar.addDescriptor(percentUnit);

  weatherService.addCharacteristic(tempChar);
  weatherService.addCharacteristic(humChar);
  weatherService.addCharacteristic(snowProbChar);

  BLE.setAdvertisedService(weatherService);

  BLE.addService(weatherService);

  // set callbacks
  //BLE.setEventHandler(BLEConnected, blePeripheralConnectHandler);
  //BLE.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);

  tempChar.writeValue(-273);
  humChar.writeValue(-1);
  snowProbChar.writeValue(-1);

  // load the model
  tflu_model = tflite::GetModel(snow_model_q_aware_tflite);

  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        tflu_model->version(), TFLITE_SCHEMA_VERSION);
    while(1);
  }

  TfLiteStatus status;

  // define ops
  static MlSnowOpResolver tflu_ops_resolver;
  status = RegisterOps(tflu_ops_resolver);

  if(status != kTfLiteOk) {
    MicroPrintf("Error registering ops (status=%d)", status);
    while(1);
  }

  // create the interpreter
  tflu_interpreter = new tflite::MicroInterpreter(tflu_model, tflu_ops_resolver, 
      tensor_arena, tensor_arena_size);

  // Allocate memory from tensor_arena for the model's tensors.
  status = tflu_interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed (status=%d)", status);
    while(1);
  }

  // get pointers to input and output tensors
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

  // Get the quantization parameters for the input and output tensors
  const auto *i_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);
  const auto *o_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_o_tensor->quantization.params);

  /* Since both input and output tensors adopt a per-tensor quantization,
   each array stores a single value. */
  tflu_i_scale = i_quantization->scale->data[0];
  tflu_i_zero_point = i_quantization->zero_point->data[0];

  tflu_o_scale = o_quantization->scale->data[0];
  tflu_o_zero_point = o_quantization->zero_point->data[0];

  // Populate temp and hum with initial values
  for (int i = 0; i < NUM_HOURS; ++i){
    float t, h;
    t = HTS.readTemperature();
    h = HTS.readHumidity();
    cur_t = t;
    cur_h = h;
    scale_and_quantize(t, h);
    t_vals[NUM_HOURS - i] = t;
    h_vals[NUM_HOURS - i] = h;
    delay(1000);  
  }

  // if(osThreadSetPriority(rtos::ThisThread::get_id(), osPriorityRealtime) != osOK){
  //   Serial.println("Error setting the priority of the main thread");
  //   while(1);
  // }

  /* setup watchdog timer to reset the system if there has been no connection
   for 20 connection periods */
  mbed::Watchdog &watchdog = mbed::Watchdog::get_instance();
  watchdog.start(20*10*1000);

  BLE.advertise();
  Serial.println("BLE WeatherStation advertising");

  // start event queue in separate thread
  eventThread.start(callback(&queue, &events::EventQueue::dispatch_forever));
  //eventThread.set_priority(osPriorityBelowNormal6);

  // generate event every 10s
  queue.call_every(10s, readPredictSend);

  //Kernel::attach_idle_hook(loop);
}


void loop() {
  BLEDevice central = BLE.central();

  // if a central is connected to the peripheral:
  if (central) {
    Serial.println("Connected to central");
    // print the central's BT address:
    //Serial.println(central.address());

    while (central.connected()) {
      //refresh the watchdog timer
      mbed::Watchdog::get_instance().kick();
      long currentMillis = millis();
      // if 20s have passed, check the sensor readings:
      if (currentMillis - previousMillis >= 20000) {
        // Critical section start
        mutex.lock();
        Serial.println("Transmitting data...");
        if(old_t != cur_t){
          tempChar.writeValue((int)cur_t);
          old_t = cur_t;
        }
        if(old_h != cur_h){
          humChar.writeValue((int)cur_h);
          old_h = cur_h;
        }        
        if(pred_f != old_pred){
          snowProbChar.writeValue(int(100*pred_f));
          old_pred = pred_f;
        }
        previousMillis = currentMillis;
        mutex.unlock();
        // critical section end
      }
    }
    Serial.println("Disconnected from central");
    //Serial.println(central.address());
  }
}


void readPredictSend(){
  const int idx1 = (cur_idx - 1 + NUM_HOURS) % NUM_HOURS;
  const int idx2 = (cur_idx - 2 + NUM_HOURS) % NUM_HOURS;

  int8_t pred_int8;
  float t_quant, h_quant;
  float t = 0.0f, h = 0.0f;

  for(int i = 0; i < num_reads; ++i){
    t += HTS.readTemperature();
    h += HTS.readHumidity();
    rtos::ThisThread::sleep_for(1s);
  }

  t /= (float)num_reads;
  h /= (float)num_reads;

  t_quant = t;
  h_quant = h;
  scale_and_quantize(t_quant, h_quant);
  t_vals[cur_idx] = t_quant;
  h_vals[cur_idx] = h_quant;

  // input observed data to model
  tflu_i_tensor->data.int8[0] = t_vals[idx2];
  tflu_i_tensor->data.int8[1] = t_vals[idx1];
  tflu_i_tensor->data.int8[2] = t_vals[cur_idx];
  tflu_i_tensor->data.int8[3] = h_vals[idx2];
  tflu_i_tensor->data.int8[4] = h_vals[idx1];
  tflu_i_tensor->data.int8[5] = h_vals[cur_idx];
  // Run inference
  tflu_interpreter->Invoke();
  // get prediction
  pred_int8 = tflu_o_tensor->data.int8[0];

  // Critical section start
  mutex.lock();
  cur_idx = (cur_idx + 1) % NUM_HOURS;
  cur_t = t;
  cur_h = h;
  // dequantize
  pred_f = (pred_int8 - tflu_o_zero_point) * tflu_o_scale;
  Serial.print("Temperature = ");
  Serial.println(t);
  Serial.print("Humidity = ");
  Serial.println(h);
  Serial.print(100*pred_f);
  Serial.println("% chance it will snow");
  mutex.unlock();
  // Critical section end
}

// void blePeripheralConnectHandler(BLEDevice central) {
//   // central connected event handler
//   isConnected = true;
//   Serial.print("Device connected, central: ");
//   Serial.println(central.address());
// }

// void blePeripheralDisconnectHandler(BLEDevice central) {
//   // central disconnected event handler
//   isConnected = false;
//   Serial.print("Device disconnected, central: ");
//   Serial.println(central.address());
// }