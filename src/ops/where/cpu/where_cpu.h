#ifndef __CPU_WHERE_H__
#define __CPU_WHERE_H__

#include "operators.h"

struct WhereCpuDescriptor{
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t output_size;
    uint64_t  *output_shape;
    uint64_t *x_strides;
    uint64_t *y_strides;
    uint64_t *condition_strides;
    uint64_t *output_indices;
};
typedef struct WhereCpuDescriptor *WhereCpuDescriptor_t;

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t handle,
WhereCpuDescriptor_t *desc_ptr,
infiniopTensorDescriptor_t condition_desc,
infiniopTensorDescriptor_t x_desc,
infiniopTensorDescriptor_t y_desc,
infiniopTensorDescriptor_t output);

// infiniopStatus_t cpuGetWhereWorkspaceSize(WhereCpuDescriptor_t desc,uint64_t *size);

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
void* condition,void* x,void* y,void *output,void* stream);

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc);



#endif