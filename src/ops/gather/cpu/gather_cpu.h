#ifndef __CPU_GATHER_H__
#define __CPU_GATHER_H__
#include "operators.h"
struct GatherCpuDescriptor{
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t indices_ndim;
    uint64_t x_ndim;
    uint64_t *x_strides;
    uint64_t *indices_strides;
    uint64_t * x_shape;
    uint64_t * indices_shape;
    uint64_t  *output_shape;
    int axis=0;
};
typedef struct GatherCpuDescriptor *GatherCpuDescriptor_t;
infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t handle,
GatherCpuDescriptor_t *desc_ptr,
infiniopTensorDescriptor_t x,
infiniopTensorDescriptor_t indices,
infiniopTensorDescriptor_t output,
int axis=0);
infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,void*x,void* indices,void* output,void* stream);
infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc);

#endif