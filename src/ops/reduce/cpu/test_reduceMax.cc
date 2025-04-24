#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>  
#include <limits>  // 需要引入limits头文件
#include <stdlib.h>
#include <stdint.h>

// 定义 Reduce 结构体
struct ReduceCpuDescriptor {
    uint64_t ndim;
    uint64_t x_shape[4];
    uint64_t y_shape[2];
    int reduce_mode;
};
typedef ReduceCpuDescriptor* ReduceCpuDescriptor_t;
inline uint64_t compactToFlat(uint64_t const *indices, uint64_t const *strides, uint64_t ndim) {
    return std::inner_product(indices, indices + ndim, strides, uint64_t(0));
}
inline uint64_t* flatToCompact(uint64_t flat_index, const uint64_t* strides, const uint64_t* shape, uint64_t ndim) {
    
    uint64_t *indices = (uint64_t*)calloc(ndim, sizeof(uint64_t));
    for (uint64_t i = 0; i < ndim; ++i) {
        indices[i] = (flat_index / strides[i]) % shape[i]; // 计算当前维度索引
        flat_index %= strides[i]; // 计算剩余索引
    }
    return indices;
}
// 计算张量的总大小
uint64_t getTotalSize(const uint64_t* shape, uint64_t ndim) {
    uint64_t size = 1;
    for (uint64_t i = 0; i < ndim; ++i) {
        size *= shape[i];
    }
    return size;
}

uint64_t* compute_strides(const uint64_t* y_shape, size_t ndim) {
    if (ndim == 0) return NULL;

    uint64_t* y_strides = (uint64_t*)malloc(ndim * sizeof(uint64_t));
    if (!y_strides) return NULL; // 分配失败

    y_strides[ndim - 1] = 1; // 最后一个维度的步长为 1
    for (int i = ndim - 2; i >= 0; --i) {
        y_strides[i] = y_strides[i + 1] * y_shape[i + 1];
    }
    
    return y_strides;
}

void dfs(const int* x, const uint64_t* x_shape, uint64_t x_ndim,
    const std::vector<uint64_t>& reduce_axes, uint64_t n_axes,
    std::vector<uint64_t>& x_id, int64_t offset, uint64_t depth) {

// 递归终止条件：所有规约轴遍历完成
if (depth == n_axes) {
   x_id.push_back(x[offset]);  // 存入当前数据
   return;
}

uint64_t axis = reduce_axes[depth];  // 规约轴索引
uint64_t stride = 1;

// 计算当前轴的步长
for (uint64_t j = axis + 1; j < x_ndim; ++j) {
   stride *= x_shape[j];
}

// 遍历当前 axis 轴上的所有值
for (uint64_t k = 0; k < x_shape[axis]; ++k) {
   dfs(x, x_shape, x_ndim, reduce_axes, n_axes, x_id, offset + k * stride, depth + 1);
}
}

// 判断 n 是否在 axes 数组中
bool is_in_axes(uint64_t n, const int64_t* axes, size_t n_axes) {
    for (size_t i = 0; i < n_axes; ++i) {
        if (axes[i] == n) {
            return true;  // 找到了，返回 true
        }
    }
    return false;  // 没找到，返回 false
}
// Reduce 操作
void reduce(ReduceCpuDescriptor_t desc, int* y, const int* x, 
            const int64_t* axes, uint64_t n_axes, int keepdims, int noop_with_empty_axes) {
    uint64_t ndim=2;
    uint64_t x_ndim=4;
    uint64_t x_shape[4] = {2, 3, 4, 5};
    uint64_t y_shape[2] = {2,  5};
    // 初始化输出张量 y
    std::fill(y, y + getTotalSize(y_shape, ndim), 0);
    uint64_t y_total_size=getTotalSize(y_shape, ndim);
    uint64_t x_total_size=getTotalSize(x_shape,ndim);
    std::vector<uint64_t> reduce_axes; // 要 reduce 的轴索引
    std::vector<uint64_t> x_id;
    uint64_t* y_strides=compute_strides(y_shape,ndim);
    uint64_t* x_strides=compute_strides(x_shape,x_ndim);
    

    // 处理负索引，转换成正索引
    for (uint64_t i = 0; i < n_axes; ++i) {
        int64_t axis = axes[i];
      
        if (axis < 0) {
            axis += x_ndim;  // 负数索引转换
        }
        if (axis >= 0 && static_cast<uint64_t>(axis) < x_ndim) {
            reduce_axes.push_back(static_cast<uint64_t>(axis));
        } else {
            std::cerr << "Error: Invalid axis " << axes[i] << std::endl;
            return;
        }
    } 
    if (n_axes == 0) {
        if (noop_with_empty_axes == 1) {
            std::copy(x, x + getTotalSize(y_shape, ndim), y);
            return;
        }
    }
    // 只实现 reduce_max
    if (desc->reduce_mode == 0) {
        if (keepdims == 0) {
           

            for (size_t i = 0; i < y_total_size; ++i) {
                x_id.clear();
                
                size_t y_index = i;

                size_t x_index = i;
                

                //1.y_indices
                uint64_t* y_indices=flatToCompact(i,y_strides,y_shape,ndim);
                //2.offset
                uint64_t offset=0;
                uint64_t flag=0;
                for(size_t i=0;i<x_ndim;i++){
                    if(!is_in_axes(i,axes,n_axes)){
                        
                        if(flag<ndim){
                            offset+=y_indices[flag]*x_strides[i];

                        }
                        flag++;
                        
                    }
                }
        
                dfs(x, x_shape, x_ndim,reduce_axes, n_axes, x_id, offset, 0);
        
                // 计算最大值
                y[y_index] = *max_element(x_id.begin(),x_id.end());
            }
        }
    }
}

// 递归打印张量
void printTensorRecursive(const std::vector<int>& tensor, const uint64_t* shape,
    uint64_t ndim, uint64_t dim, uint64_t& index) {
std::cout << "[";
if (dim == ndim - 1) {  // 递归到最后一维，直接打印数据
for (uint64_t i = 0; i < shape[dim]; ++i) {
std::cout << tensor[index++];
if (i < shape[dim] - 1) std::cout << ", ";
}
} else {  // 递归处理高维
for (uint64_t i = 0; i < shape[dim]; ++i) {
printTensorRecursive(tensor, shape, ndim, dim + 1, index);
if (i < shape[dim] - 1) std::cout << ", ";
}
}
std::cout << "]";
}

// 打印张量入口
void printTensor(const std::vector<int>& tensor, const uint64_t* shape, uint64_t ndim) {
uint64_t index = 0;
printTensorRecursive(tensor, shape, ndim, 0, index);
std::cout << std::endl;
}

int main() {
    // 你的 x 数据
    std::vector<int> x = {
        52, 48, 80, 46, 55, 30, 79, 5, 53, 90, 91, 81, 19, 36, 81, 36, 56, 33, 81, 6,  
        79, 41, 53, 54, 96, 80, 96, 95, 74, 8, 51, 54, 73, 34, 12, 94, 93, 70, 91, 38,  
        86, 22, 55, 59, 20, 100, 21, 9, 97, 34, 33, 42, 17, 67, 99, 22, 63, 19, 61, 63,  
        57, 44, 71, 5, 30, 36, 73, 23, 67, 98, 46, 9, 37, 66, 97, 33, 32, 68, 42, 16,  
        2, 10, 10, 50, 71, 49, 15, 78, 19, 36, 61, 72, 60, 100, 44, 73, 47, 16, 51, 98,  
        79, 95, 16, 29, 48, 37, 61, 83, 2, 16, 91, 17, 4, 25, 18, 19, 61, 16, 16, 62  
    };

    uint64_t x_shape[4] = {2, 3, 4, 5};
    uint64_t y_shape[2] = {2, 5}; // 对第 1 维 reduce_max
    

    // 归约轴
    int64_t axes[] = {1, 2};
    uint64_t n_axes = 2;

    // 生成 y
    std::vector<int> y(getTotalSize(y_shape, 2));

    // 初始化 ReduceCpuDescriptor_t 指针
    ReduceCpuDescriptor_t desc = new ReduceCpuDescriptor;
    desc->ndim = 4;
    std::copy(std::begin(x_shape), std::end(x_shape), desc->x_shape);
    std::copy(std::begin(y_shape), std::end(y_shape), desc->y_shape);
    desc->reduce_mode = 0;  // Max

    // 执行 reduce 操作
    reduce(desc, y.data(), x.data(), axes, n_axes, 0, 0);
    


    // 输出结果
    std::cout << "Input Tensor X:\n";
    printTensor(x, x_shape,4);
    std::cout << "\nReduced Tensor Y (max along axis 1,2):\n";
    printTensor(y, y_shape,2);

    // 释放分配的内存
    delete desc;

    return 0;
}
/*
x:
tensor([[[[ 52,  48,  80,  46,  55],
          [ 30,  79,   5,  53,  90],
          [ 91,  81,  19,  36,  81],
          [ 36,  56,  33,  81,   6]],

         [[ 79,  41,  53,  54,  96],
          [ 80,  96,  95,  74,   8],
          [ 51,  54,  73,  34,  12],
          [ 94,  93,  70,  91,  38]],

         [[ 86,  22,  55,  59,  20],
          [100,  21,   9,  97,  34],
          [ 33,  42,  17,  67,  99],
          [ 22,  63,  19,  61,  63]]],


        [[[ 57,  44,  71,   5,  30],
          [ 36,  73,  23,  67,  98],
          [ 46,   9,  37,  66,  97],
          [ 33,  32,  68,  42,  16]],

         [[  2,  10,  10,  50,  71],
          [ 49,  15,  78,  19,  36],
          [ 61,  72,  60, 100,  44],
          [ 73,  47,  16,  51,  98]],

         [[ 79,  95,  16,  29,  48],
          [ 37,  61,  83,   2,  16],
          [ 91,  17,   4,  25,  18],
          [ 19,  61,  16,  16,  62]]]])
After reduce along first dimension:
tensor([[[[ 86,  48,  80,  59,  96],
          [100,  96,  95,  97,  90],
          [ 91,  81,  73,  67,  99],
          [ 94,  93,  70,  91,  63]]],


        [[[ 79,  95,  71,  50,  71],
          [ 49,  73,  83,  67,  98],
          [ 91,  72,  60, 100,  97],
          [ 73,  61,  68,  51,  98]]]])
After reduce along second dimension:
tensor([[[[100,  96,  95,  97,  99]]],


        [[[ 91,  95,  83, 100,  98]]]])
Final result after removing extra dimensions:
tensor([[100,  96,  95,  97,  99],
        [ 91,  95,  83, 100,  98]])

*/