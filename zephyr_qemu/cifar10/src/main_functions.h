#ifndef TINYML_COOKBOOK_CIFAR10_QEMU
#define TINYML_COOKBOOK_CIFAR10_QEMU

/* Expose a C friendly interface for main functions. */
#ifdef __cplusplus
extern "C" {
#endif

/* Initializes all data needed for the example. The name is important, and needs
 * to be setup() for Arduino compatibility.
 */
void setup(void);

/* Runs one iteration of data gathering and inference. This should be called
 * repeatedly from the application code. The name needs to be loop() for Arduino
 * compatibility.
 */
void loop(void);

#ifdef __cplusplus
}
#endif

#endif /* TINYML_COOKBOOK_CIFAR10_QEMU */
