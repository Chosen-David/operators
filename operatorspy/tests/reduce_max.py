from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64,c_int
import ctypes
import sys
import os
import time
import random
from typing import List, Optional
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

class ReduceMaxDescriptor(Structure):
     
    _fields_ = [("device", c_int32)]


infiniopReduceMaxDescriptor_t = POINTER(ReduceMaxDescriptor)


def reduce(x, axes,
    keepdims,
    noop_with_empty_axes,
    reduce_type):
    reduce_layers = {
        0: torch.amax,
        1: torch.mean,
        2: torch.min,
    }
    reduce_type=0

    if reduce_type not in reduce_layers:
        raise ValueError(f"Invalid reduce_type: {reduce_type}. Supported: 0 (max), 1 (mean), 2 (min)")

     # 如果 axes 为空，并且 noop_with_empty_axes=True，则直接返回 x
    if axes is None and noop_with_empty_axes:
        return x

    reduction_fn = reduce_layers[reduce_type]

    if reduce_type == 2:  # max and min return tuple (values, indices)
        result = reduction_fn(x, dim=axes, keepdim=keepdims)[0]
    else:
        result = reduction_fn(x, dim=axes, keepdim=keepdims)

    return result


def inferShape(
    x_shape: List[int],
    axes: Optional[List[int]] = None,
    keepdims: bool = False,
    noop_with_empty_axes: bool = False
) -> List[int]:
    """

    """
    if axes is None:  # No axes specified
        if noop_with_empty_axes:
            return x_shape  # No operation performed
        return [1] if not keepdims else [1] * len(x_shape)  # Reduce over all dims

    # Normalize negative axes (e.g., -1 refers to last dim)
    axes = [a if a >= 0 else len(x_shape) + a for a in axes]

    if keepdims:
        # Keep reduced dimensions as 1
        return [1 if i in axes else x_shape[i] for i in range(len(x_shape))]
    else:
        # Remove reduced dimensions
        return [x_shape[i] for i in range(len(x_shape)) if i not in axes]
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)


def test(
    lib,
    handle,
    torch_device,
    x_shape, 
    axes, 
    n_axes,
    keepdims,
    noop_with_empty_axes,
    reduce_type,
    tensor_dtype=torch.float16,
):
    reduce_type=0
    # Determine the reduce type string based on reduce_type value
    if reduce_type == 0:
        reduce_name = "ReduceMax"
    elif reduce_type == 1:
        reduce_name = "ReduceMean"
    elif reduce_type == 2:
        reduce_name = "ReduceMin"
    else:
        reduce_name = "UnknownReduce"

    print(
        f"Testing {reduce_name} on {torch_device} with x_shape: {x_shape}, "
        f"axes: {axes}, n_axes: {n_axes}, keepdims: {keepdims}, "
        f"noop_with_empty_axes: {noop_with_empty_axes}, dtype: {tensor_dtype}"
    )


    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.rand(inferShape(x_shape, axes,keepdims,noop_with_empty_axes), dtype=tensor_dtype).to(torch_device)
    
    ans=reduce(x,axes,keepdims,noop_with_empty_axes,reduce_type)

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)


    descriptor = infiniopReduceMaxDescriptor_t()

    check_error(
        lib.infiniopCreateReduceMaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            tuple_to_void_p(axes),
            n_axes,
            keepdims,
            noop_with_empty_axes,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    # workspaceSize = ctypes.c_uint64(0)
    # check_error(
    #     lib.infiniopGetReduceMaxWorkspaceSize(descriptor, ctypes.byref(workspaceSize))
    # )
    print("Input:", x)
    print("indices:", axes)
    check_error(lib.infiniopReduceMax(descriptor,y_tensor.data,x_tensor.data,None))
    print("Output:", y)
    print("Expected Answer:", ans)



    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyReduceMaxDescriptor(descriptor))
    print("释放over")


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape,axes,n_axes,keepdims,noop_with_empty_axes,reduce_type in test_cases:
        reduce_type=0
        test(lib, handle, "cpu",x_shape,axes,n_axes,keepdims,noop_with_empty_axes,reduce_type,tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape,axes,n_axes,keepdims,noop_with_empty_axes,reduce_type, tensor_dtype=torch.float32)
    print("开始销毁handle")
    destroy_handle(lib, handle)
    print("handle已经销毁")
if __name__ == "__main__":
    # Define test cases
    test_cases = [
        # Test case 1: ReduceMax on a 2D tensor
        ((4, 4), [1], 1, True, False, 0),  # x_shape: (4, 4), axes: [1], reduce_type: ReduceMax
        
        # Test case 2: ReduceMean on a 3D tensor
        ((3, 3,3), [0, 1], 2, False, True, 1),  # x_shape: (3, 3, 3), axes: [0, 1], reduce_type: ReduceMean
        
        # Test case 3: ReduceMin on a 4D tensor
        ((2, 4, 4, 4), [2], 1, True, False, 2),  # x_shape: (2, 4, 4, 4), axes: [2], reduce_type: ReduceMin
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateReduceMaxDescriptor.restype=c_int32
    lib.infiniopCreateReduceMaxDescriptor.argtypes=[
        infiniopHandle_t,
        POINTER(infiniopReduceMaxDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_uint64,
        c_int,
        c_int,
    ]
    lib.infiniopReduceMax.restype=c_int32
    lib.infiniopReduceMax.argtypes=[
        infiniopReduceMaxDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyReduceMaxDescriptor.restype=c_int32
    lib.infiniopDestroyReduceMaxDescriptor.argtypes=[
        infiniopReduceMaxDescriptor_t,
    ]
    if args.cpu:
        test_cpu(lib,test_cases)
    else:
        test_cpu(lib,test_cases)
    
    print("\033[92mTest passed!\033[0m")

