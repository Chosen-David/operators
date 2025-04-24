// #include "../../../devices/cuda/common_cuda.h"
// #include "../../utils.h"

// // Kernel for clipping F32 tensor
// template <typename T>
// __global__ void clipKernel(T *y, const T *x, T min_val, T max_val, int64_t size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         y[idx] = min(max(x[idx], min_val), max_val);
//     }
// }

// template <typename T>
// void clipCuda(T *y, const T *x, T min_val, T max_val, int64_t size, cudaStream_t stream) {
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

//     clipKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(y, x, min_val, max_val, size);
//     checkCudaError(cudaGetLastError());
// }

// // Wrapper for clipping function
// infiniopStatus_t cudaClip(void *y, const void *x, float min_val, float max_val, int64_t size, void *stream, int dtype) {
//     if (dtype == F32) {
//         clipCuda<float>((float *)y, (const float *)x, min_val, max_val, size, (cudaStream_t)stream);
//     } else if (dtype == F16) {
//         clipCuda<half>((half *)y, (const half *)x, (half)min_val, (half)max_val, size, (cudaStream_t)stream);
//     } else {
//         return STATUS_BAD_TENSOR_DTYPE;
//     }
//     return STATUS_SUCCESS;
// }
