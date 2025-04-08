#ifndef __CPU_REDUCE_H__
#define __CPU_REDUCE_H__

#include "../../../devices/cpu/common_cpu.h"
#include "operators.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>



struct ReduceCpuDescriptor {
    Device device;
    DataLayout dt;
    uint64_t y_size;
    uint64_t x_ndim;
    uint64_t ndim;
    uint64_t const *y_shape;
    uint64_t const *x_shape;
    int64_t const *y_strides;
    int64_t const *x_strides;
    int64_t const *axes;
    uint64_t n_axes;
    int keepdims;
    int noop_with_empty_axes;
    int reduce_mode;
};

typedef struct ReduceCpuDescriptor *ReduceCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
                                           ReduceCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x,
                                           int64_t const *axes,
                                           uint64_t n_axes,
                                           int keepdims,
                                           int noop_with_empty_axes,
                                           int reduce_type);

// infiniopStatus_t cpuGetReduceWorkspaceSize(ReduceCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                           void *y,
                           void const *x,
                           void *stream);

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc);

#endif
