from ctypes import POINTER, Structure, c_int32, c_void_p
import ctypes
import sys
import os

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


class GatherDescriptor(Structure):
    _fields_ = [("device", c_int32), ("axis", c_int32)]


infiniopGatherDescriptor_t = POINTER(GatherDescriptor)


import numpy as np

def gather(data, indices, axis=0):
    """
    Gathers elements from `data` along the specified `axis` using `indices`.
    
    :param data: np.ndarray, input tensor of rank r >= 1
    :param indices: np.ndarray, tensor of indices (int32 or int64)
    :param axis: int, axis along which to gather
    :return: np.ndarray, gathered output tensor
    """
    data = np.asarray(data)
    indices = np.asarray(indices)
    
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Indices must be of integer type")
    
    if axis < 0:
        axis += data.ndim  # Handle negative axis
    
    if axis < 0 or axis >= data.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for data with {data.ndim} dimensions")
    
    return np.take(data, indices, axis=axis)

# Example usage
data = np.array([
    [1.0, 1.2, 1.9],
    [2.3, 3.4, 3.9],
    [4.5, 5.7, 5.9]
])
indices = np.array([[0, 2]])
axis = 1

output = gather(data, indices, axis)
print(output)

def test(
    lib,
    handle,
    torch_device,
    x_shape,
    indices_shape,
    output_shape,
    axis,
    tensor_dtype=torch.float16,
):
    print(
        f"Testing Gather on {torch_device} with x_shape:{x_shape} indices_shape:{indices_shape} output_shape:{output_shape} axis:{axis} dtype:{tensor_dtype}"
    )

    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    indices = torch.randint(0, x.shape[axis], indices_shape, dtype=torch.int64).to(torch_device)
    output = torch.rand(output_shape, dtype=tensor_dtype).to(torch_device)

    ans = gather(x, indices, axis)

    x_tensor = to_tensor(x, lib)
    indices_tensor = to_tensor(indices, lib)
    output_tensor = to_tensor(output, lib)
    descriptor = infiniopGatherDescriptor_t()
    print(f"Debug: axis={axis}, x_shape={x_shape}, x_ndim={len(x_shape)}")
    # Print input information
   
 
    
    print("x (input tensor):")
    print(x)

    print("indices (input indices):")
    print(indices)


    check_error(
        lib.infiniopCreateGatherDescriptor(
            handle,
            ctypes.byref(descriptor),
            x_tensor.descriptor,
            indices_tensor.descriptor,
            output_tensor.descriptor,
            axis,
        )
    )

    x_tensor.descriptor.contents.invalidate()
    indices_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()

    check_error(lib.infiniopGather(descriptor, x_tensor.data, indices_tensor.data, output_tensor.data, None))
    print("Output:", output)
    print("Expected Answer:", ans)
    

    assert torch.allclose(output, torch.tensor(ans, dtype=output.dtype, device=output.device), atol=0, rtol=0)
    


    check_error(lib.infiniopDestroyGatherDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, indices_shape, output_shape, axis in test_cases:
        test(lib, handle, "cpu", x_shape, indices_shape, output_shape, axis, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape, indices_shape, output_shape, axis, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":

    test_cases = [
        # (x_shape, indices_shape, output_shape, axis)
        ((3, 3), (1, 2), (3, 1, 2), 1),
        ((4, 5), (2, 3), (2, 3, 5), 0),
        ((2, 3, 4), (2, 2), (2, 2, 3, 4), 0),
        ((3, 4, 5), (2,), (3, 2, 5), 1),
        ((5,), (3,), (3,), 0),
    ]

    args = get_args()
    lib = open_lib()

    # 设定 gather 算子函数签名
    lib.infiniopCreateGatherDescriptor.restype = c_int32
    lib.infiniopCreateGatherDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGatherDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]

    lib.infiniopGather.restype = c_int32
    lib.infiniopGather.argtypes = [
        infiniopGatherDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyGatherDescriptor.restype = c_int32
    lib.infiniopDestroyGatherDescriptor.argtypes = [
        infiniopGatherDescriptor_t,
    ]

    # 运行测试
    if args.cpu:
        test_cpu(lib, test_cases)

    if not (args.cpu):
        test_cpu(lib, test_cases)

    print("\033[92mTest passed!\033[0m")
