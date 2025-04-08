#include "../reduce/reduce.h"
#include "../utils.h"
#include "ops/reduce_min/reduce_min.h"

struct _ReduceMinDescriptor {
    Device device;
    infiniopReduceDescriptor_t reduce_desc;
};
typedef struct _ReduceMinDescriptor *_ReduceMinDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMinDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMinDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const *axes,
    uint64_t n_axes,
    int keepdims,
    int noop_with_empty_axes) {
    
    infiniopReduceDescriptor_t reduce_desc;
    std::cout << "Creating infiniopCreateReduceMinDescriptor" << std::endl;
    CHECK_STATUS(
        infiniopCreateReduceDescriptor(handle, &reduce_desc, y, x, axes, n_axes, keepdims, noop_with_empty_axes, 2), 
        STATUS_SUCCESS);
    std::cout << "Creating..." << std::endl;
    *(_ReduceMinDescriptor_t *)desc_ptr = new _ReduceMinDescriptor{
        handle->device,
        reduce_desc};
    std::cout << "Created!" << std::endl;
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopReduceMin(
    infiniopReduceMinDescriptor_t desc, void *y, void const *x, void *stream) {
    auto _desc = (_ReduceMinDescriptor_t)desc;
    CHECK_STATUS(infiniopReduce(_desc->reduce_desc, y, x, stream), STATUS_SUCCESS);
    return STATUS_SUCCESS;
}

__C __export infiniopStatus_t infiniopDestroyReduceMinDescriptor(
    infiniopReduceMinDescriptor_t desc) {
    CHECK_STATUS(infiniopDestroyReduceDescriptor(((_ReduceMinDescriptor_t)desc)->reduce_desc), STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}
