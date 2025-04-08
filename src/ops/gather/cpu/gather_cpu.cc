#include "gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include "data_type.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <functional> // std::multiplies

void replaceAtIndex(std::vector<uint64_t>& A, int index, const std::vector<uint64_t>& B) {
    if (index >= A.size()) {
        std::cerr << "Error: Index out of range!\n";
        return;
    }

    // 删除 A[index]
    A.erase(A.begin() + index);

    // 在删除的位置插入 B 的所有元素
    A.insert(A.begin() + index, B.begin(), B.end());
}
infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t handle,
    GatherCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t indices,
    infiniopTensorDescriptor_t output,
    int axis){
    if (output->dt != x->dt){
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (!is_contiguous(x) || !is_contiguous(indices)||!is_contiguous(output)){
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (output->dt != F16 && output->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    
    uint64_t ndim=output->ndim;
    uint64_t x_ndim=x->ndim;
    uint64_t indices_ndim=indices->ndim;
    uint64_t output_size=std::accumulate(output->shape, output->shape + output->ndim, 1ULL, std::multiplies<uint64_t>());
    std::vector<uint64_t> A(x->shape, x->shape + x->ndim);
    std::vector<uint64_t> B(indices->shape,indices->shape+indices->ndim);
    replaceAtIndex(A,axis,B);
    for(size_t i=0;i<ndim;i++){
        if(output->shape[i]!=A[i]){
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
       
    uint64_t *indices_strides = new uint64_t[indices_ndim];  // 假设 ndim 是你想要复制的维度数
    uint64_t *indices_shape = new uint64_t[indices_ndim];
    std::copy(indices->shape, indices->shape + indices_ndim, indices_shape);


    std::copy(indices->strides, indices->strides + indices_ndim, indices_strides);

    uint64_t *x_strides = new uint64_t[x_ndim];  // 假设 ndim 是你想要复制的维度数

    std::copy(x->strides, x->strides + x_ndim, x_strides);
    uint64_t *x_shape=new uint64_t[x_ndim];
    std::copy(x->shape, x->shape + x_ndim, x_shape);


    uint64_t  *output_shape=new uint64_t[ndim];
    std::copy(output->shape,output->shape+ndim,output_shape);
    // std::cout<<"check axis....."<<std::endl;
    // std::cout << "axis: " << axis << ", x_ndim: " << x_ndim << std::endl;
    


    
    // Ensure axis is within the valid range [-x_ndim, x_ndim)
    if (axis < -static_cast<int64_t>(x_ndim) || axis >= static_cast<int64_t>(x_ndim)) {
        return STATUS_BAD_PARAM;
    }
    
    // std::cout << "axis ok" << std::endl;
    

    
    

    *desc_ptr=new GatherCpuDescriptor{
        DevCpu,
        output->dt,
        ndim,
        indices_ndim,
        x_ndim,
        x_strides,
        indices_strides,
        x_shape,
        indices_shape,
        output_shape,
        axis,
    };
    std::cout<<"created!"<<std::endl;
    return STATUS_SUCCESS;
}
infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc){
    delete[] desc->indices_strides;
    delete[] desc->output_shape;
    delete[] desc->x_strides;
    delete desc;
    return STATUS_SUCCESS;
}
// 处理 std::vector 版本
inline std::vector<uint64_t> flatToCompact(uint64_t flat_index, 
    const std::vector<uint64_t>& strides, 
    const std::vector<uint64_t>& shape) {
    uint64_t ndim = shape.size();
    std::vector<uint64_t> indices(ndim, 0);

    for (uint64_t i = 0; i < ndim; ++i) {
    indices[i] = (flat_index / strides[i]) % shape[i]; // 计算当前维度索引
    flat_index %= strides[i]; // 计算剩余索引
    }

    return indices;
}

// 处理 uint64_t* 数组的版本
inline uint64_t* flatToCompact(uint64_t flat_index, 
const uint64_t* strides, 
const uint64_t* shape, 
uint64_t ndim) {
    uint64_t* indices = (uint64_t*)calloc(ndim, sizeof(uint64_t));

    for (uint64_t i = 0; i < ndim; ++i) {
    indices[i] = (flat_index / strides[i]) % shape[i];
    flat_index %= strides[i];
    }

    return indices;
}

// 处理 std::vector<uint64_t> 的情况
inline uint64_t getTotalSize(const std::vector<uint64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<uint64_t>());
}

// // 处理 uint64_t* 数组的情况
// inline uint64_t getTotalSize(const uint64_t* arr, uint64_t ndim) {
//     return std::accumulate(arr, arr + ndim, 1ULL, std::multiplies<uint64_t>());
// }
inline uint64_t compactToFlat(const std::vector<uint64_t>& indices, const std::vector<uint64_t>& strides) {
    return std::inner_product(indices.begin(), indices.end(), strides.begin(), uint64_t(0));
}
std::vector<uint64_t> compute_strides(const std::vector<uint64_t>& y_shape) {
    size_t ndim = y_shape.size();
    if (ndim == 0) return {}; // 空形状返回空 strides

    std::vector<uint64_t> y_strides(ndim, 1); // 预分配 strides 数组
    for (int i = ndim - 2; i >= 0; --i) {
        y_strides[i] = y_strides[i + 1] * y_shape[i + 1];
    }

    return y_strides;
}
template<typename Tdata>
infiniopStatus_t gather_cpu(GatherCpuDescriptor_t desc,void*x,void* indices,void* output){
    auto indices_=reinterpret_cast<int64_t *>(indices);
    auto x_=reinterpret_cast<Tdata *>(x);
    auto output_=reinterpret_cast<Tdata *>(output);
    uint64_t indices_ndim=desc->indices_ndim;
    uint64_t x_ndim=desc->x_ndim;
    uint64_t ndim=desc->ndim;
    auto axis=desc->axis;
    // 使用 vector 代替原始指针
    std::vector<uint64_t> x_shape(desc->x_shape, desc->x_shape + x_ndim);
    std::vector<uint64_t> output_shape(desc->output_shape, desc->output_shape + ndim);
    std::vector<uint64_t> indices_shape(desc->indices_shape, desc->indices_shape + indices_ndim);
    // 计算 strides
    uint64_t output_total_size = getTotalSize(output_shape);
    uint64_t n_indices=getTotalSize(indices_shape);
    std::vector<uint64_t> output_strides = compute_strides(output_shape);
    std::vector<uint64_t> x_strides = compute_strides(x_shape);
    std::vector<uint64_t> indices_strides=compute_strides(indices_shape);
    std::vector<uint64_t> gather_axes; // indices转换后
    // 处理负索引，转换成正索引
    if(axis<0){
        axis+=x_ndim;
    }
    for (uint64_t i = 0; i < n_indices; ++i) {
        int64_t axi = indices_[i];
        if (axi < 0) {
            axi += x_shape[axis];  // 负数索引转换
        }
        if (axi >= 0 && static_cast<uint64_t>(axi) < x_shape[axis]) {
            gather_axes.push_back(static_cast<uint64_t>(axi));
        } else {
            std::cerr << "Error: Invalid axis " <<axi << std::endl;
            return STATUS_BAD_PARAM;
        }
    } 
    std::vector<uint64_t> x_ids;
    for(size_t i=0;i<output_total_size;i++){

        //1.output_ids
        std::vector<uint64_t> output_ids = flatToCompact((i), output_strides, output_shape);
        //2.对应的x_ids
        x_ids.clear();
        uint64_t stride=0;
        for(size_t j=0;j<output_ids.size();j++){
            
            if(j==desc->axis){
                for(int k=0;k<indices_shape.size();k++){
                    stride+=indices_strides[k]*output_ids[j+k];
                }
                x_ids.push_back(indices_[stride]);
                j=j+indices_shape.size();
            }
            if(j<output_ids.size()){
                x_ids.push_back(output_ids[j]);
            }
        }
        //3.填入
        auto x_index=compactToFlat(x_ids,x_strides);
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            output_[i]=f32_to_f16(f16_to_f32(x_[x_index]));

        }
        else{
            output_[i]=x_[x_index];
        }
    }

    return STATUS_SUCCESS;

}
infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,void*x,void* indices,void* output,void* stream){
    if(desc->dtype==F16){
        return gather_cpu<uint16_t>(desc,x,indices,output);

    }
    if(desc->dtype==F32){
        return gather_cpu<float>(desc,x,indices,output);

    }
    return STATUS_BAD_TENSOR_DTYPE;
    

}
