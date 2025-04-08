#include "clip_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t handle,
                                         ClipCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x,
                                         float min_val,
                                         float max_val) {
    if (y->ndim != x->ndim || y->dt != x->dt) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    uint64_t* x_shape=new uint64_t[x->ndim];
    std::copy(x->shape,x->shape+x->ndim,x_shape);
    
    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        y->dt,
        x_shape,
        y->ndim,
        min_val,
        max_val
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template <typename T>
void applyClip(ClipCpuDescriptor_t desc, T *y, T *x, uint64_t size) {
#pragma omp parallel for
    for (uint64_t i = 0; i < size; ++i) {
        // std::cout<<"x: "<<x[i]<<std::endl;
        if constexpr (std::is_same<T, uint16_t>::value) {
            y[i] = std::min(std::max(f32_to_f16(f16_to_f32(x[i])), static_cast<T>(desc->min_val)), static_cast<T>(desc->max_val));

        }
        else{
            y[i] = std::min(std::max(x[i], static_cast<T>(desc->min_val)), static_cast<T>(desc->max_val));

        }
        
    }
}

template <typename T>
infiniopStatus_t clip_cpu(ClipCpuDescriptor_t desc, void *y, void *x, uint64_t size) {
    std::cout<<"entering applyClip"<<std::endl;
    auto x_=reinterpret_cast<T *>(x);
    auto y_=reinterpret_cast<T *>(y);
    for (uint64_t i = 0; i < size; ++i) {
        
        
        if constexpr (std::is_same<T, uint16_t>::value) {
            // if(i<10){
            //     std::cout<<"x: "<<f16_to_f32(x_[i])<<std::endl;
    
            // }
            // auto min=(desc->min_val);
            // auto max=(desc->max_val);

            // std::cout<<"min:"<<min<<std::endl;
            // std::cout<<"max:"<<max<<std::endl;

            y_[i] = f32_to_f16(std::min(
                std::max(static_cast<float>((f16_to_f32(x_[i]))), static_cast<float>(((desc->min_val)))), 
                static_cast<float>(((desc->max_val)))
            ));
            // y_[i]=std::min(std::max(static_cast<T>(x_[i]), static_cast<float>(desc->min_val)), static_cast<float>(desc->max_val));
            


        }
        else{
            y_[i] = std::min(std::max(static_cast<float>(x_[i]), static_cast<float>(desc->min_val)), static_cast<float>(desc->max_val));

        }
        
    }
    
   
    // applyClip<T>(desc, y_, x_, size);
    return STATUS_SUCCESS;
}
inline uint64_t getTotalSize(const uint64_t* arr, uint64_t ndim) {
    return std::accumulate(arr, arr + ndim, 1ULL, std::multiplies<uint64_t>());
}
infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *y, void *x,
                         void *stream) {
    std::cout<<"entering clip_cpu"<<std::endl;
    uint64_t size = 1;
    size=getTotalSize(desc->x_shape,desc->ndim);


    // // 打印 ndim（张量维度）
    // std::cout << "ndim: " << desc->ndim << std::endl;

    // // 打印 x_shape（假设它是一个 int* 指向的数组）
    // std::cout << "x_shape: [";
    // for (int i = 0; i < desc->ndim; ++i) {
    //     std::cout << desc->x_shape[i];
    //     if (i < desc->ndim - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    // // 打印计算出的 size
    // std::cout << "size: " << size << std::endl;

   
    
    
 
    if (desc->dtype == F16) {
        // float* x_ptr = static_cast<float*>(x);
        // float* y_ptr = static_cast<float*>(y);
        // std::cout << "x (float): ";
        // for (uint64_t i = 0; i < size; ++i) {
        //     std::cout << x_ptr[i] << " ";
        // }
        // std::cout << std::endl;
        
        // std::cout << "y (float): ";
        // for (uint64_t i = 0; i < size; ++i) {
        //     std::cout << y_ptr[i] << " ";
        // }
        // std::cout << std::endl;

        return clip_cpu<uint16_t>(desc, y, x, size);
    }

    if (desc->dtype == F32) {
        // float* x_ptr = static_cast<float*>(x);
        // float* y_ptr = static_cast<float*>(y);
        // std::cout << "x (float): ";
        // for (uint64_t i = 0; i < size; ++i) {
        //     std::cout << x_ptr[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "y (float): ";
        // for (uint64_t i = 0; i < size; ++i) {
        //     std::cout << y_ptr[i] << " ";
        // }
        // std::cout << std::endl;

        return clip_cpu<float>(desc, y, x, size);
    }
    
    return STATUS_BAD_TENSOR_DTYPE;
}
