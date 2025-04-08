#include "../reduce/reduce.h"

#include "../utils.h"
#include "ops/reduce_max/reduce_max.h"

struct _ReduceMaxDescriptor{
    Device device;
    infiniopReduceDescriptor_t reduce_desc;

};
typedef struct _ReduceMaxDescriptor *_ReduceMaxDescriptor_t;


__C __export infiniopStatus_t infiniopCreateReduceMaxDescriptor(infiniopHandle_t handle,
infiniopReduceMaxDescriptor_t *desc_ptr,
infiniopTensorDescriptor_t y,
infiniopTensorDescriptor_t x,
int64_t const *axes,
uint64_t n_axes,
int keepdims,
int noop_with_empty_axes){
    infiniopReduceDescriptor_t reduce_desc;
    std::cout<<"creating infiniopCreateReduceDescriptor"<<std::endl;
    CHECK_STATUS(infiniopCreateReduceDescriptor(handle,&reduce_desc,y,x,axes,n_axes,keepdims,noop_with_empty_axes,0), STATUS_SUCCESS);
    std::cout<<"creating ....."<<std::endl;
    *(_ReduceMaxDescriptor_t *) desc_ptr = new _ReduceMaxDescriptor{
        handle->device,
        reduce_desc};
    std::cout<<"created!!!!!"<<std::endl;
    return STATUS_SUCCESS;

}

__C __export infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc,void *y, void const *x, void *stream) {
    auto _desc=(_ReduceMaxDescriptor_t) desc;
    CHECK_STATUS(infiniopReduce(_desc->reduce_desc,y,x,stream),STATUS_SUCCESS);
    std::cout<<"reduceMax结束"<<std::endl;
    return STATUS_SUCCESS;


}

__C __export infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc){
    CHECK_STATUS(infiniopDestroyReduceDescriptor(((_ReduceMaxDescriptor_t) desc)->reduce_desc),STATUS_SUCCESS);
    delete desc;
    return STATUS_SUCCESS;
}