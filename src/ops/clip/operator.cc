#include "../utils.h"
#include "operators.h"
#include "ops/clip/clip.h"

#ifdef ENABLE_CPU
#include "cpu/clip_cpu.h"
#endif


__C  infiniopStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    float min_val,
    float max_val) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateClipDescriptor(handle, (ClipCpuDescriptor_t *) desc_ptr, y, x, min_val, max_val);
#endif

    }
    return STATUS_BAD_DEVICE;
}

__C  infiniopStatus_t infiniopClip(
    infiniopClipDescriptor_t desc,
    void *y,
    void *x,
    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            std::cout<<"entering cpuClip"<<std::endl;
    
            
            return cpuClip((ClipCpuDescriptor_t) desc, y, x, stream);
#endif

    }
    return STATUS_BAD_DEVICE;
}

__C  infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyClipDescriptor((ClipCpuDescriptor_t) desc);
#endif

    }
    return STATUS_BAD_DEVICE;
}
