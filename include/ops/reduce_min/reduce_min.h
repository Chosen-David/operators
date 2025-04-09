#ifndef REDUCE_MIN_H
#define REDUCE_MIN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMinDescriptor {
    Device device;
} ReduceMinDescriptor;
typedef ReduceMinDescriptor* infiniopReduceMinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMinDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMinDescriptor_t* desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const* axes,
    uint64_t n_axes,
    int keepdims,
    int noop_with_empty_axes
);

__C __export infiniopStatus_t infiniopGetReduceMinWorkspaceSize(
    infiniopReduceMinDescriptor_t desc,
    uint64_t* size
);

__C __export infiniopStatus_t infiniopReduceMin(
    infiniopReduceMinDescriptor_t desc,
    void* workspace,
    uint64_t workspace_size,
    void* y,
    void const* x,
    void* stream
);

__C __export infiniopStatus_t infiniopDestroyReduceMinDescriptor(
    infiniopReduceMinDescriptor_t desc
);

#endif // REDUCE_MIN_H
