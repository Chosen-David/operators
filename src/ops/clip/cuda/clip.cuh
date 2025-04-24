// #ifndef __CUDA_CLIP_H__
// #define __CUDA_CLIP_H__

// #include "../../../devices/cuda/common_cuda.h"
// #include "../../../devices/cuda/cuda_handle.h"
// #include "operators.h"

// struct ClipCudaDescriptor {
//     Device device;
//     DT dtype;
//     int device_id;
//     float min_val;
//     float max_val;
//     uint64_t size;
// };

// typedef struct ClipCudaDescriptor *ClipCudaDescriptor_t;

// infiniopStatus_t cudaCreateClipDescriptor(CudaHandle_t handle,
//                                           ClipCudaDescriptor_t *desc_ptr,
//                                           infiniopTensorDescriptor_t y,
//                                           infiniopTensorDescriptor_t x,
//                                           float min_val,
//                                           float max_val);

// infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
//                           void *y,
//                           void const *x,
//                           void *stream);

// infiniopStatus_t cudaDestroyClipDescriptor(ClipCudaDescriptor_t desc);

// #endif // __CUDA_CLIP_H__