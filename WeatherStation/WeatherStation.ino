#include <Arduino_HTS221.h>
#include <Arduino_LPS22HB.h>
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
BLEIntCharacteristic pressureChar("2AA3", BLERead | BLENotify);
BLEIntCharacteristic snowProbChar("2A78", BLERead | BLENotify); // Displayed as Rainfall

unsigned long long previousMillis;  // last time data was transmitted

constexpr int NUM_HOURS = 3;  // period of sensing and prediction
constexpr int num_reads = 3;  // number of times to repeat sensor reading
int8_t t_vals[NUM_HOURS], h_vals[NUM_HOURS]; // quantized values
// store current readings and readings previously transmitted
float cur_t, cur_h, cur_p, pred_f;
float old_t = -273, old_h = -1, old_p = -1, old_pred = -1;

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

void readPredict();

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
    1. Create README
    2. Update NN model to use barometric pressure
  */
  tflite::InitializeTarget();   // Initializes serial communication

  if(!HTS.begin()){
    Serial.println("Failed initializations of HTS221!");
    while(1);
  }

  if (!BARO.begin()) {
    Serial.println("Failed to initialize pressure sensor!");
    while (1);
  }

  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("WeatherStation");

  BLEDescriptor celsiusUnit("272F", "°C");
  BLEDescriptor percentUnit("27AD", "%");

  tempChar.addDescriptor(celsiusUnit);
  humChar.addDescriptor(percentUnit);
  snowProbChar.addDescriptor(percentUnit);

  weatherService.addCharacteristic(tempChar);
  weatherService.addCharacteristic(humChar);
  weatherService.addCharacteristic(pressureChar);
  weatherService.addCharacteristic(snowProbChar);

  BLE.setAdvertisedService(weatherService);

  BLE.addService(weatherService);

  // set callbacks
  //BLE.setEventHandler(BLEConnected, blePeripheralConnectHandler);
  //BLE.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);

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
    float t, h, p;
    t = HTS.readTemperature()-2;
    h = HTS.readHumidity();
    p = BARO.readPressure();
    cur_t = t;
    cur_h = h;
    cur_p = p;
    scale_and_quantize(t, h);
    t_vals[NUM_HOURS - i] = t;
    h_vals[NUM_HOURS - i] = h;
    delay(1000);  
  }

  // if(osThreadSetPriority(rtos::ThisThread::get_id(), osPriorityRealtime) != osOK){
  //   Serial.println("Error setting the priority of the main thread");
  //   while(1);
  // }

  // setup watchdog timer to reset the system if it gets stuck
  mbed::Watchdog &watchdog = mbed::Watchdog::get_instance();
  watchdog.start(2*60*1000);

  BLE.advertise();
  Serial.println("BLE WeatherStation advertising");

  // start event queue in separate thread
  eventThread.start(callback(&queue, &events::EventQueue::dispatch_forever));
  //eventThread.set_priority(osPriorityBelowNormal6);

  // generate event every 60s
  queue.call_every(60s, readPredict);

  //Kernel::attach_idle_hook(loop);
}


void loop() {
  BLEDevice central = BLE.central();
  bool firstTime = true;
  //refresh the watchdog timer
  mbed::Watchdog::get_instance().kick();

  // if a central is connected to the peripheral:
  if (central) {
    long currentMillis = millis();
    previousMillis = millis();
    Serial.println("Connected to central");
    // print the central's BT address:
    //Serial.println(central.address());

    while (central.connected()) {
      mbed::Watchdog::get_instance().kick();
      currentMillis = millis();
      // check the for new sensor readings every 20s
      if (currentMillis - previousMillis >= 20000 | firstTime) {
        // Critical section start
        mutex.lock();
        Serial.println("Transmitting data...");
        if(old_t != cur_t | firstTime){
          tempChar.writeValue((int)cur_t);
          old_t = cur_t;
        }
        if(old_h != cur_h | firstTime){
          humChar.writeValue((int)cur_h);
          old_h = cur_h;
        }
        if(old_p != cur_p | firstTime){
          pressureChar.writeValue((int)cur_p);
          old_p = cur_p;
        }        
        if(pred_f != old_pred | firstTime){
          snowProbChar.writeValue(int(100*pred_f));
          old_pred = pred_f;
        }
        previousMillis = currentMillis;
        mutex.unlock();
        // critical section end
      }
      firstTime = false;
    }
    Serial.println("Disconnected from central");
    //Serial.println(central.address());
  }
}


void readPredict(){
  const int idx1 = (cur_idx - 1 + NUM_HOURS) % NUM_HOURS;
  const int idx2 = (cur_idx - 2 + NUM_HOURS) % NUM_HOURS;

  int8_t pred_int8;
  float t_quant, h_quant;
  float t = 0.0f, h = 0.0f, p = 0.0f;

  for(int i = 0; i < num_reads; ++i){
    t += HTS.readTemperature()-2; // account for error
    h += HTS.readHumidity();
    p += BARO.readPressure();
    rtos::ThisThread::sleep_for(5s);
  }

  t /= (float)num_reads;
  h /= (float)num_reads;
  p /= (float)num_reads;

  t_quant = round(t);
  h_quant = round(h);
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
  cur_p = p;
  // dequantize
  pred_f = (pred_int8 - tflu_o_zero_point) * tflu_o_scale;
  MicroPrintf("Temperature = %d deg. C", (int)t);
  MicroPrintf("Humidity = %d\%", (int)h);
  MicroPrintf("Barometric pressure = %d kPa", (int)p);
  MicroPrintf("%d\% chance it will snow", (int)(100*pred_f));
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