#ifndef REDUCE_MEAN_H
#define REDUCE_MEAN_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMeanDescriptor {
    Device device;
} ReduceMeanDescriptor;

typedef ReduceMeanDescriptor *infiniopReduceMeanDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMeanDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMeanDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const *axes,
    uint64_t n_axes,
    int keepdims,
    int noop_with_empty_axes);

__C __export infiniopStatus_t infiniopGetReduceMeanWorkspaceSize(
    infiniopReduceMeanDescriptor_t desc,
    uint64_t *size);

__C __export infiniopStatus_t infiniopReduceMean(
    infiniopReduceMeanDescriptor_t desc,
    void *workspace,
    uint64_t workspace_size,
    void *y,
    void const *x,
    void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMeanDescriptor(
    infiniopReduceMeanDescriptor_t desc);

#endif // REDUCE_MEAN_H
