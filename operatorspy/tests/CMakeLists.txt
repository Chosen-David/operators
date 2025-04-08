cmake_minimum_required(VERSION 3.10)
project(TestWhereCPU)
set(CONDA_PREFIX "/home/yswang/miniconda3")

# 设置 C++ 版本
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 指定 Google Test 的路径（使用 Conda 安装的路径）
set(CONDA_PREFIX $ENV{CONDA_PREFIX})

# 添加 Google Test
find_package(GTest REQUIRED PATHS ${CONDA_PREFIX} NO_DEFAULT_PATH)
include_directories(${GTEST_INCLUDE_DIRS})

# 添加项目的 include 目录
include_directories(${CONDA_PREFIX}/include)
include_directories(/home/yswang/winter_learn/ai_compiler/operators/include)

# 指定源文件
set(SOURCE_FILES /home/yswang/winter_learn/ai_compiler/operators/src/ops/where/cpu/test.cc)

# 生成可执行文件
add_executable(test_where_cpu ${SOURCE_FILES})

# 链接 Google Test 和 pthread
target_link_libraries(test_where_cpu GTest::GTest GTest::Main pthread)

# 启用测试
enable_testing()
add_test(NAME TestWhereCPU COMMAND test_where_cpu)
