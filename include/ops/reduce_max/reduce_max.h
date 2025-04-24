#ifndef REDUCE_MAX_H
#define REDUCE_MAX_H
#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMaxDescriptor{
    Device device;

}ReduceMaxDescriptor;
typedef ReduceMaxDescriptor *infiniopReduceMaxDescriptor_t;


__C __export infiniopStatus_t infiniopCreateReduceMaxDescriptor(infiniopHandle_t handle,
    infiniopReduceMaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const *axes,
    uint64_t n_axes,
    int keepdims,
    int noop_with_empty_axes);


__C __export infiniopStatus_t infiniopGetReduceMaxWorkspaceSize(infiniopReduceMaxDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc,
    void *workspace, uint64_t workspace_size,
    void *y,
    void const *x,
    void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc);

#endif