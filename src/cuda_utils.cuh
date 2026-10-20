/**
 * @file cuda_utils.cuh
 * @brief CUDA error checking macros and CUDA event timing utilities.
 *
 * Course: CSC14120 - Parallel Programming (Final Project)
 * Authors: Group XX
 */

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err));                     \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

/**
 * @brief Helper class for accurate GPU kernel timing using CUDA events.
 */
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
#endif // __CUDACC__

#endif // CUDA_UTILS_CUH
