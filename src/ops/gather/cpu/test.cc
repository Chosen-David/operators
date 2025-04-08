#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>  // for memset
#include "../../../devices/cpu/common_cpu.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <functional> // std::multiplies

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

// 结构体定义
struct GatherCpuDescriptor {
    uint64_t axis;
    uint64_t ndim;
    uint64_t x_ndim;
    uint64_t indices_ndim;
    uint64_t x_shape[2];  // 假设最多 4 维
    uint64_t indices_shape[2];
    uint64_t output_shape[3];
};
typedef GatherCpuDescriptor* GatherCpuDescriptor_t;

// 测试用 gather_cpu
template<typename Tdata>
void gather_cpu(GatherCpuDescriptor_t desc,void*x,void* indices,void* output){
    auto indices_=reinterpret_cast<int64_t *>(indices);
    auto x_=reinterpret_cast<Tdata *>(x);
    auto output_=reinterpret_cast<Tdata *>(output);
    
    uint64_t indices_ndim=desc->indices_ndim;
    uint64_t x_ndim=desc->x_ndim;
    uint64_t ndim=desc->ndim;
 
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
    for (uint64_t i = 0; i < n_indices; ++i) {
        int64_t axi = indices_[i];
        if (axi < 0) {
            axi += x_shape[desc->axis];  // 负数索引转换
        }
        if (axi >= 0 && static_cast<uint64_t>(axi) < x_shape[desc->axis]) {
            gather_axes.push_back(static_cast<uint64_t>(axi));
        } else {
            std::cerr << "Error: Invalid axis " <<axi << std::endl;
            return ;
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

    return;

}

// 测试函数
// 测试函数
void test_gather_cpu() {
    using Tdata = float;  // 测试 float 类型的数据

    // 输入张量x（4x5）
    std::vector<float> x = {
        0.7197, 0.1553, 0.5420, 0.7520, 0.8184,
        0.8110, 0.8682, 0.2568, 0.3027, 0.6802,
        0.3032, 0.1030, 0.3853, 0.7075, 0.2554,
        0.3828, 0.2993, 0.8457, 0.5640, 0.1216
    };

    // 输入索引 (2x3)
    std::vector<int64_t> indices = {1, 0, 1, 1, 1, 0};

    // 输出张量（shape: 2x3x5）
    std::vector<float> output(30, 0.0f);

    // GatherCpuDescriptor 配置
    GatherCpuDescriptor desc;
    desc.axis = 0;
    desc.ndim = 3; // output rank
    desc.x_ndim = 2;
    desc.indices_ndim = 2;

    uint64_t x_shape[2] = {4, 5};      // x shape: (4, 5)
    uint64_t indices_shape[2] = {2, 3}; // indices shape: (2, 3)
    uint64_t output_shape[3] = {2, 3, 5}; // output shape: (2, 3, 5)

    std::memcpy(desc.x_shape, x_shape, sizeof(x_shape));
    std::memcpy(desc.indices_shape, indices_shape, sizeof(indices_shape));
    std::memcpy(desc.output_shape, output_shape, sizeof(output_shape));

    // 执行 gather 操作
    gather_cpu<Tdata>(&desc, x.data(), indices.data(), output.data());

    // 打印输出结果
    std::cout << "Gather Output:\n";
    for (size_t i = 0; i < output.size(); i++) {
        std::cout << output[i] << " ";
        if ((i + 1) % 5 == 0) std::cout << std::endl;
    }
}

int main() {
    test_gather_cpu();
    return 0;
}


/*
x (input tensor):
tensor([[0.7197, 0.1553, 0.5420, 0.7520, 0.8184],
        [0.8110, 0.8682, 0.2568, 0.3027, 0.6802],
        [0.3032, 0.1030, 0.3853, 0.7075, 0.2554],
        [0.3828, 0.2993, 0.8457, 0.5640, 0.1216]], dtype=torch.float16)
indices (input indices):
tensor([[1, 0, 1],
        [1, 1, 0]])



Gather Output:
[[[0.811  0.8682 0.2568 0.3027 0.6802]
  [0.7197 0.1553 0.542  0.752  0.8184]
  [0.811  0.8682 0.2568 0.3027 0.6802]]

 [[0.811  0.8682 0.2568 0.3027 0.6802]
  [0.811  0.8682 0.2568 0.3027 0.6802]
  [0.7197 0.1553 0.542  0.752  0.8184]]]
*/