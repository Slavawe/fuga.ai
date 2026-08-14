// cuda_stubs.c — Заглушки для CUDA-функций (для тестирования без реальной CUDA)
// Компилируется через: gcc -c -fPIC cuda_stubs.c -o cuda_stubs.o
//                      ar rcs libcuda_stubs.a cuda_stubs.o
// Линкуется: rustc ... -L. -lcuda_stubs

#include <stdint.h>

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NO_DEVICE 100

// Все CUDA-функции возвращают CUDA_ERROR_NO_DEVICE (нет устройства)
// Это позволяет коду компилироваться и работать с GpuContext.available=false

uint32_t cuInit(uint32_t flags) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuDeviceGet(int32_t* device, int32_t ordinal) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuCtxCreate(uint64_t* pctx, uint32_t flags, int32_t device) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuModuleLoadData(uint64_t* module, const uint8_t* image) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuModuleGetFunction(uint64_t* hfunc, uint64_t hmod, const char* name) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuMemAlloc(uint64_t* dptr, uint64_t bytesize) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuMemcpyHtoD(uint64_t dst, const uint8_t* src, uint64_t byte_count) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuMemcpyDtoH(uint8_t* dst, uint64_t src, uint64_t byte_count) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuMemFree(uint64_t dptr) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuLaunchKernel(
    uint64_t f,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t shared_mem_bytes,
    uint64_t hstream,
    uint8_t** kernel_params,
    uint8_t** extra
) {
    return CUDA_ERROR_NO_DEVICE;
}

uint32_t cuCtxSynchronize() {
    return CUDA_ERROR_NO_DEVICE;
}
