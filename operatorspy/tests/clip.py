from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)
from operatorspy.tests.test_utils import get_args

import torch
from typing import Tuple

PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class ClipDescriptor(Structure):
    _fields_ = [
        ("device", c_int32),

    ]


infiniopClipDescriptor_t = POINTER(ClipDescriptor)

def test(lib, handle, torch_device, x_shape, min_val, max_val, tensor_dtype=torch.float16):
    print(
        f"Testing Clip on {torch_device} with x_shape: {x_shape}, min_val: {min_val}, max_val: {max_val}, dtype: {tensor_dtype}"
    )
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.zeros_like(x)

    ans = torch.clamp(x, min=min_val, max=max_val)
   
    
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopClipDescriptor_t()
    


    check_error(
        lib.infiniopCreateClipDescriptor(
            handle, ctypes.byref(descriptor), y_tensor.descriptor, x_tensor.descriptor, min_val, max_val
        )
    )
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    print("Input tensor x (first 10 elements):", x.flatten()[:10])  # 打印 x 数据的前 10 个数

    # print("x_tensor.data:", x_tensor.data)  # Printing x_tensor's data

    
  
    check_error(lib.infiniopClip(descriptor, y_tensor.data, x_tensor.data, None))
    
    # if PROFILE:
    #     start_time = time.time()
    #     for i in range(NUM_ITERATIONS):
    #         check_error(lib.infiniopClip(descriptor, y_tensor.data, x_tensor.data, None))
    #     elapsed = (time.time() - start_time) / NUM_ITERATIONS
    #     print(f"    lib time: {elapsed :6f}")
    print("Computed output:", y)
    print("Expected output:", ans)

    
    assert torch.allclose(y, ans, atol=0, rtol=1e-2)
    check_error(lib.infiniopDestroyClipDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, min_val, max_val in test_cases:
        test(lib, handle, "cpu", x_shape, min_val, max_val, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape, min_val, max_val, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


# def test_cuda(lib, test_cases):
#     device = DeviceEnum.DEVICE_CUDA
#     handle = create_handle(lib, device)
#     for x_shape, min_val, max_val in test_cases:
#         test(lib, handle, "cuda", x_shape, min_val, max_val, tensor_dtype=torch.float16)
#         test(lib, handle, "cuda", x_shape, min_val, max_val, tensor_dtype=torch.float32)
#     destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, min_val, max_val
        ((32, 3, 4), 0.1, 0.9),
        ((1, 3, 4, 4), -0.5, 0.5),
        ((32, 3, 128, 128), 0.0, 1.0),
        ((1, 1, 4, 4, 4), -1.0, 1.0),
        ((32, 3, 32, 32, 32), -0.2, 0.8),
    ]
    args = get_args()
    
    lib = open_lib()
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopClipDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopClipDescriptor_t,
    ]

    test_cpu(lib, test_cases)
    # test_cuda(lib, test_cases)
    print("\033[92mTest passed!\033[0m")