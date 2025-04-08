#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__

#include "../../../devices/cpu/common_cpu.h"
#include "operators.h"
#include <algorithm>
#include <cstring>
#include <numeric>

struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t* x_shape;
    uint64_t ndim;
    float min_val;
    float max_val;
};

typedef struct ClipCpuDescriptor *ClipCpuDescriptor_t;

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t handle,
                                         ClipCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x,
                                         float min_val,
                                         float max_val);

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *y, void *x,
                         void *stream);

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);

#endif