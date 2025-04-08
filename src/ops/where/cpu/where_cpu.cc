#include "where_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include "data_type.h"
#include <cmath>

inline void incrementOne(uint64_t *indices, uint64_t const *shape, uint64_t ndim) {
    for (int64_t i = ndim - 1; i >= 0; --i) {
        if (++indices[i] != shape[i]) {
            return;
        }
        indices[i] = 0;
    }
}

inline uint64_t compactToFlat(uint64_t const *indices, uint64_t const *strides, uint64_t ndim) {
    return std::inner_product(indices, indices + ndim, strides, uint64_t(0));
}
infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t handle, WhereCpuDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t condition_desc, infiniopTensorDescriptor_t x_desc, 
                                          infiniopTensorDescriptor_t y_desc,infiniopTensorDescriptor_t output) {
                                            
    // // 检查 condition 是否是布尔类型的矩阵
    // if (condition_desc->dt != BOOL) {
    //     return STATUS_BAD_TENSOR_DTYPE;  // condition 必须是布尔类型
    // }
    
    if (!is_contiguous(x_desc) || !is_contiguous(condition_desc) || !is_contiguous(y_desc)||!is_contiguous(output)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (output->dt != F16 && output->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (output->dt != x_desc->dt || output->dt != y_desc->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    // // Check tensor dimensionalities
    // if (condition_desc->ndim != x_desc->ndim || condition_desc->ndim != y_desc->ndim) {
    //     return STATUS_BAD_TENSOR_SHAPE;  // 确保维度一致
    // }
    
    uint64_t ndim=output->ndim;
    uint64_t output_size=std::accumulate(output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t *x_strides=new uint64_t[ndim];
    uint64_t *y_strides=new uint64_t[ndim];
    uint64_t *condition_strides=new uint64_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        x_strides[i] = (i < ndim - x_desc->ndim || output->shape[i] != x_desc->shape[i + x_desc->ndim - ndim]) ? 0 : x_desc->strides[i + x_desc->ndim - ndim];
        y_strides[i] = (i < ndim - y_desc->ndim || output->shape[i] != y_desc->shape[i + y_desc->ndim - ndim]) ? 0 : y_desc->strides[i + y_desc->ndim - ndim];
        condition_strides[i]= (i < ndim - condition_desc->ndim || output->shape[i] != condition_desc->shape[i + condition_desc->ndim - ndim]) ? 0 : condition_desc->strides[i + condition_desc->ndim - ndim];
    }
    uint64_t *output_indices=new uint64_t[ndim];
    std::fill(output_indices,output_indices+ndim,0);
    uint64_t *output_shape = new uint64_t[ndim];
    std::copy(output->shape, output->shape + ndim, output_shape);
    *desc_ptr=new WhereCpuDescriptor{
        DevCpu,
        output->dt,
        ndim,
        output_size,
        output_shape,
        x_strides,
        y_strides,
        condition_strides,
        output_indices,    
    };
    return STATUS_SUCCESS;
}

// infiniopStatus_t cpuGetWhereWorkspaceSize(WhereCpuDescriptor_t desc, uint64_t *size) {
//     *size = 0;
//     return STATUS_SUCCESS;
// }

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc){
    delete[] desc->output_shape;
    delete[] desc->x_strides;
    delete[] desc->y_strides;
    delete[] desc->condition_strides;
    delete[] desc->output_indices;
    delete desc;
    return STATUS_SUCCESS;
}
template<typename Tdata>
infiniopStatus_t where_cpu(WhereCpuDescriptor_t desc, void *condition, void *x, void *y, void *output) {
    auto condition_ = reinterpret_cast<uint8_t *>(condition);
    auto x_ = reinterpret_cast<Tdata *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    auto output_ = reinterpret_cast<Tdata *>(output);
    auto &indices=desc->output_indices;

    for (uint64_t i = 0; i < desc->output_size; ++i, incrementOne(indices, desc->output_shape, desc->ndim)) {
        auto x_index = compactToFlat(indices, desc->x_strides, desc->ndim);
        auto y_index = compactToFlat(indices, desc->y_strides, desc->ndim);
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            output_[i] = condition_[i] ? f32_to_f16(f16_to_f32(x_[x_index])) : f32_to_f16(f16_to_f32(y_[y_index]));

        } else {

            output_[i] = condition_[i] ? x_[x_index] : y_[y_index];
        }

    }


    return STATUS_SUCCESS;
}

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
    void* condition,void* x,void* y,void *output,void* stream){
        if(desc->dtype==F16){
            return where_cpu<uint16_t>(desc,condition,x,y,output);
        }
        if (desc->dtype == F32) {
            return where_cpu<float>(desc, condition,x,y,output);
        }
        return STATUS_BAD_TENSOR_DTYPE;

    
}