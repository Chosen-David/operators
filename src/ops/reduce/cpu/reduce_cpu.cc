#include "reduce_cpu.h"
#include "../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
#include <iostream>
#include <vector>
#include <cassert>



// get the total number of elements in arr
inline uint64_t getTotalSize(const uint64_t *arr, uint64_t ndim) {
    return std::accumulate(arr, arr + ndim, 1ULL, std::multiplies<uint64_t>());
}

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t handle,
    ReduceCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int64_t const *axes,
    uint64_t n_axes,
    int keepdims,
    int noop_with_empty_axes,
    int reduce_type){
        std::cout<<"begin reduce creating"<<std::endl;
        if(reduce_type>2){
            return STATUS_BAD_PARAM;
        }
        
        if (y->dt != F16 && y->dt != F32) {
            return STATUS_BAD_TENSOR_DTYPE;
        }
        if (y->dt != x->dt) {
            return STATUS_BAD_TENSOR_DTYPE;
        }
        std::cout<<"111  "<<reduce_type<<std::endl;
        auto x_ndim=x->ndim;
        auto ndim=y->ndim;
       
        std::cout<<"ndim: "<<ndim<<std::endl;
        // std::cout<<"n_axis: "<<n_axes<<std::endl;
        for (size_t i = 0; i < n_axes; ++i) {
            int64_t p = axes[i];  // 保证 p 的类型为 int64_t
            std::cout<<"p: "<<p<<std::endl;
            if (p >= static_cast<int64_t>(x_ndim) || p < -static_cast<int64_t>(x_ndim)) {
                return STATUS_BAD_PARAM;
            }
        }
        
        std::cout<<"333"<<std::endl;

        const auto y_size=getTotalSize(y->shape,ndim);
        uint64_t *x_shape=new uint64_t[x_ndim];
        uint64_t *y_shape=new uint64_t[ndim];
        int64_t *y_strides=new int64_t[ndim];
        int64_t *x_strides=new int64_t[x_ndim];
       
        int64_t *axes_=new int64_t[n_axes];
        memcpy(x_shape, x->shape, x_ndim * sizeof(uint64_t));
        memcpy(y_shape, y->shape, ndim * sizeof(uint64_t));
        memcpy(x_strides, x->strides, x_ndim * sizeof(int64_t));
        memcpy(y_strides, y->strides, ndim * sizeof(int64_t));
        for (size_t i = 0; i < n_axes; ++i) {
            axes_[i]=axes[i];
        }

        
        std::cout<<"creating CPU"<<std::endl;
        *desc_ptr=new ReduceCpuDescriptor{
            DevCpu,
            y->dt,
            y_size,
            x_ndim,
            ndim,
            y_shape,
            x_shape,
            y_strides,
            x_strides,
            axes_,
            n_axes,
            keepdims,
            noop_with_empty_axes,   
            reduce_type,

        };

        return STATUS_SUCCESS;
}




infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    std::cout << "进入cpuDestroyReduceDescriptor" << std::endl;
    if (desc) {
        if (desc->x_shape) {
            std::cout << "释放 x_shape" << std::endl;
            delete[] desc->x_shape;
            desc->x_shape = nullptr;
        }

        if (desc->y_shape) {
            std::cout << "释放 y_shape" << std::endl;
            delete[] desc->y_shape;
            desc->y_shape = nullptr;
        }

        if (desc->axes) {
            std::cout << "释放 axes" << std::endl;
            delete[] desc->axes;
            desc->axes = nullptr;
        }
 
        if (desc->y_strides) {
            std::cout << "释放 y_strides" << std::endl;
            delete[] desc->y_strides;
            desc->y_strides = nullptr;
        }
        
        if (desc->x_strides) {
            std::cout << "释放 x_strides" << std::endl;
            delete[] desc->x_strides;
            desc->x_strides = nullptr;
        }
        

        std::cout << "释放 desc" << std::endl;
        delete desc;
    }

    std::cout << "释放完毕" << std::endl;


    return STATUS_SUCCESS;
}

// inline uint64_t* flatToCompact(uint64_t flat_index, const int64_t* strides, const uint64_t* shape, uint64_t ndim) {
    
//     uint64_t *indices = new uint64_t[ndim];
//     for (uint64_t i = 0; i < ndim; ++i) {
//         indices[i] = (flat_index / strides[i]) % shape[i]; // 计算当前维度索引
//         flat_index %= strides[i]; // 计算剩余索引
//     }
//     return indices;
// }
inline std::vector<uint64_t> flatToCompact(uint64_t flat_index, const int64_t* strides, const uint64_t* shape, uint64_t ndim) {
    std::vector<uint64_t> indices(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        indices[i] = (flat_index / strides[i]) % shape[i]; // 计算当前维度索引
        flat_index %= strides[i]; // 更新剩余索引
    }
    return indices;
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

template<typename Xdata,typename Ydata>
void dfs(const Xdata* x, const uint64_t* x_shape, uint64_t x_ndim,
    const std::vector<uint64_t>& reduce_axes, uint64_t n_axes,
    std::vector<float>& x_id, int64_t offset, uint64_t depth) {

// 递归终止条件：所有规约轴遍历完成
if (depth == n_axes) {
 
    if constexpr (std::is_same<Xdata, uint16_t>::value) {
        float value=f16_to_f32(x[offset]);
        x_id.push_back(value);  // 存入当前数据
    }else{
        x_id.push_back(x[offset]);  // 存入当前数据
    }
   
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
   dfs<Xdata,Ydata>(x, x_shape, x_ndim, reduce_axes, n_axes, x_id, offset + k * stride, depth + 1);
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
template<typename Xdata, typename Ydata>
inline void reduce(ReduceCpuDescriptor_t desc, Ydata *y, Xdata const* x) {


        // 获取输入张量 x 和输出张量 y 的形状和维度
        auto axes=desc->axes;
        auto keepdims=desc->keepdims;
        auto n_axes=desc->n_axes;
        auto noop_with_empty_axes=desc->noop_with_empty_axes;
        
        // const uint64_t *x_shape = desc->x_shape;
        // const uint64_t *y_shape = desc->y_shape;
        uint64_t x_ndim = desc->x_ndim;
        uint64_t ndim = desc->ndim;

        // 初始化输出张量 y
        std::fill(y, y + desc->y_size, 0);
        uint64_t y_total_size=getTotalSize(desc->y_shape, ndim);
        uint64_t x_total_size=getTotalSize(desc->x_shape,x_ndim);
        std::vector<uint64_t> reduce_axes; // 要 reduce 的轴索引
        std::vector<float> x_id;
        // float* x_ids=new[x_total_size];
        // uint64_t* y_strides=compute_strides(y_shape,ndim);
        // uint64_t* x_strides=compute_strides(desc->x_shape,x_ndim);
        // const int64_t* y_strides=desc->y_strides;
        // const int64_t* x_strides=desc->x_strides;

        for (uint64_t i = 0; i < n_axes; ++i) {
            int64_t axis = axes[i];
            std::cout<<"axis "<<axis<<std::endl;
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
                std::copy(x, x + getTotalSize(desc->y_shape, ndim), y);
                return;
            }
        }
        for (size_t i = 0; i < y_total_size; ++i) {
            x_id.clear();   
            uint64_t y_index = i;
            uint64_t x_index = i;
            //1.y_indices
            //uint64_t* y_indices=flatToCompact(i,y_strides,desc->y_shape,ndim);
            std::vector<uint64_t> y_indices = flatToCompact(i, desc->y_strides, desc->y_shape, ndim);

            //2.offset
            uint64_t offset=0;
            uint64_t flag=0;
            for(size_t i=0;i<x_ndim;i++){
                if((!is_in_axes(i,axes,n_axes))||(desc->y_shape[flag]==1)){
                    if(flag<ndim){
                        offset+=y_indices[flag]*desc->x_strides[i];
                    }
                    flag++; 
                }
            }
            // delete[] y_indices;  // 在调用 flatToCompact() 的地方释放内存    
            // std::cout<<offset<<" offset"<<std::endl;

            dfs<Xdata,Ydata>(x, desc->x_shape, x_ndim,reduce_axes, n_axes, x_id, offset, 0);
            for(auto p:x_id){
                std::cout<<p<<" "<<std::endl;
            }
            std::cout<<""<<std::endl;


            std::cout<<"mode "<<desc->reduce_mode<<std::endl;
            switch (desc->reduce_mode) {
                case 0: // Max          
                    y[y_index] = *std::max_element(x_id.begin(),x_id.end());    
                    break;
                case 1://Mean
                    y[y_index]=std::accumulate(x_id.begin(), x_id.end(), 0ULL)/x_id.size();
                    break;
                case 2://Min
                    y[y_index]=*std::min_element(x_id.begin(), x_id.end());
                    break;

            }
                 
        }
        std::cout<<"结束 "<<std::endl;


        // free(y_strides);
        // free(x_strides);
       
}

// 检查轴是否为空（即轴的大小为 1 或没有有效数据）
bool isEmptyAxis(uint64_t *shape, uint64_t ndim, int64_t const *axes, uint64_t n_axes) {
    for (uint64_t i = 0; i < n_axes; ++i) {
        int axis = axes[i];
        if (axis < 0) axis += ndim; // 处理负数轴
        if (shape[axis] == 1) {
            return true;  // 如果某个轴的大小为 1，则认为该轴为空
        }
    }
    return false;
}

template <typename T>
void print_type(const T& x) {
    std::cout << "Data type of x: " << typeid(x).name() << std::endl;
}

template<typename Tdata>
infiniopStatus_t reduce_cpu(ReduceCpuDescriptor_t desc,void* y,void const *x){
    auto y_=reinterpret_cast<Tdata*>(y);
    auto x_ = reinterpret_cast<Tdata const *>(x);
    std::fill(y_, y_ + desc->y_size, 0);
   
    reduce<Tdata,Tdata>(desc, y_,  x_);
    return STATUS_SUCCESS; 
}
template<>
infiniopStatus_t reduce_cpu<uint16_t>(ReduceCpuDescriptor_t desc, void* y, void const *x) {
    
    auto y_ = reinterpret_cast<float*>(y);
    auto x_ = reinterpret_cast<uint16_t const*>(x);

    print_type(x_);
    
    uint64_t x_total_size = getTotalSize(desc->x_shape, desc->x_ndim);  // Calculate x total size


    std::fill(y_, y_ + desc->y_size, 0);

    // 调用 FP32 规约函数
    reduce<uint16_t, float>(desc, y_, x_);


     //将 FP32 结果转换回 FP16
     auto y_16 = reinterpret_cast<uint16_t*>(y);
     #pragma omp parallel for
     for (size_t i = 0; i < desc->y_size; ++i) {
         y_16[i] = f32_to_f16(y_[i]);  // 需要一个 float32 -> float16 的转换函数
     }
 

    std::cout <<"reduce 结束" <<std::endl;

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
    void *y,
    void const *x,
    void *stream){
        if(desc->dt==F16){
            return reduce_cpu<uint16_t>(desc,y,x);
  
        }
        if(desc->dt==F32){
            return reduce_cpu<float>(desc,y,x);
        
        }
        return STATUS_BAD_TENSOR_DTYPE;
}

