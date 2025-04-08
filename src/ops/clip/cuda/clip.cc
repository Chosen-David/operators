// #include "clip.cuh""
// #include "../../../devices/cuda/common_cuda.h"
// #include "../../utils.h"

// __global__ void clipKernel(float *y, const float *x, float min_val, float max_val, size_t size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         y[idx] = fminf(fmaxf(x[idx], min_val), max_val);
//     }
// }

// infiniopStatus_t cudaCreateClipDescriptor(
//     CudaHandle_t handle,
//     ClipCudaDescriptor_t *desc_ptr,
//     infiniopTensorDescriptor_t y,
//     infiniopTensorDescriptor_t x,
//     float min_val,
//     float max_val) {

//     if (!desc_ptr || y->ndim != x->ndim || y->dt != x->dt) {
//         return STATUS_BAD_PARAM;
//     }

//     size_t size = 1;
//     for (int i = 0; i < y->ndim; ++i) {
//         if (y->shape[i] != x->shape[i]) {
//             return STATUS_BAD_TENSOR_SHAPE;
//         }
//         size *= y->shape[i];
//     }

//     *desc_ptr = new ClipCudaDescriptor{
//         DevNvGpu,
//         y->dt,
//         handle->device_id,
//         min_val,
//         max_val,
//         size
//     };

//     return STATUS_SUCCESS;
// }

// infiniopStatus_t cudaClip(
//     ClipCudaDescriptor_t desc,
//     void *y,
//     void *x,
//     void *stream) {

//     if (!desc || !y || !x) {
//         return STATUS_BAD_PARAM;
//     }

//     float *y_d = static_cast<float *>(y);
//     const float *x_d = static_cast<const float *>(x);
//     size_t size = desc->size;

//     int blockSize = 256;
//     int gridSize = (size + blockSize - 1) / blockSize;

//     clipKernel<<<gridSize, blockSize, 0, static_cast<cudaStream_t>(stream)>>>(y_d, x_d, desc->min_val, desc->max_val, size);

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         return STATUS_EXECUTION_FAILED;
//     }

//     return STATUS_SUCCESS;
// }

// infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc) {
//     if (!desc) {
//         return STATUS_BAD_PARAM;
//     }
//     delete desc;
//     return STATUS_SUCCESS;
// }