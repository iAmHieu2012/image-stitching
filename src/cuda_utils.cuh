/**
 * @file cuda_utils.cuh
 * @brief Macro kiểm tra lỗi CUDA (CHECK) và bộ đo thời gian GPU (GpuTimer) dùng CUDA Events.
 *
 * Tham khảo: Source code demo lý thuyết của thầy Phạm Trọng Nghĩa
 */

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <stdio.h>
#include <stdlib.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>

// Macro kiểm tra lỗi trả về của các hàm CUDA Runtime API
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

// Cấu trúc đo thời gian thực thi trên GPU bằng CUDA Events
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
#endif // __CUDACC__

#endif // CUDA_UTILS_CUH
