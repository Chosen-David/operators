#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <omp.h>

// 状态返回类型
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1
} infiniopStatus_t;

// Clip 结构体
struct ClipCpuDescriptor {
    float min_val;
    float max_val;
};

using ClipCpuDescriptor_t = ClipCpuDescriptor*;

// Clip 实现
template <typename T>
void applyClip(ClipCpuDescriptor_t desc, T *y, T *x, uint64_t size) {
#pragma omp parallel for
    for (uint64_t i = 0; i < size; ++i) {
        y[i] = std::min(std::max(x[i], static_cast<T>(desc->min_val)), static_cast<T>(desc->max_val));
    }
}

template <typename T>
infiniopStatus_t clip_cpu(ClipCpuDescriptor_t desc, void *y, void *x, uint64_t size) {
    applyClip<T>(desc, reinterpret_cast<T *>(y), reinterpret_cast<T *>(x), size);
    return STATUS_SUCCESS;
}

// 测试函数
void test_clip_cpu() {
    using Tdata = float;

    // 创建测试数据
    std::vector<Tdata> input = { -2.5334, -1.0223, 0.0, 1.5444, 3.02234, 5.51233 };
    std::vector<Tdata> output(input.size(), 0.0f);

    // 定义 Clip 结构体（裁剪区间 [-1.0, 3.0]）
    ClipCpuDescriptor desc;
    desc.min_val = -1.0f;
    desc.max_val = 3.0f;

    // 执行 Clip 操作
    clip_cpu<Tdata>(&desc, output.data(), input.data(), input.size());

    // 打印结果
    std::cout << "Input Data:  ";
    for (auto v : input) std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "Clipped Data: ";
    for (auto v : output) std::cout << v << " ";
    std::cout << std::endl;
}

int main() {
    test_clip_cpu();
    return 0;
}
