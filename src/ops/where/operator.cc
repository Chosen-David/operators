#include "../utils.h"
#include "operators.h"
#include "ops/where/where.h"

#ifdef ENABLE_CPU
#include "cpu/where_cpu.h"
#endif

__C infiniopStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t condition,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t output
){
    switch(handle->device){
#ifdef ENABLE_CPU
    case DevCpu:
        std::cout<<"creat!"<<std::endl;
        return  cpuCreateWhereDescriptor(handle,(WhereCpuDescriptor_t *)desc_ptr,condition,x,y,output);
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc,void* condition,void* x,void* y,void* output,void* stream){
    switch (desc->device)
    {
#ifdef ENABLE_CPU
case DevCpu:
    return cpuWhere((WhereCpuDescriptor_t) desc, condition, x, y, output,stream);
#endif  
    }
    return STATUS_BAD_DEVICE;
}
__C infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc){
    switch (desc->device) {
        #ifdef ENABLE_CPU
                case DevCpu:
                    return cpuDestroyWhereDescriptor((WhereCpuDescriptor_t) desc);
        #endif
    }
    return  STATUS_BAD_DEVICE;
}
