from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64, c_int
import ctypes
import sys
import os
import torch
from typing import List, Optional, Tuple
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

class ReduceMinDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopReduceMinDescriptor_t = POINTER(ReduceMinDescriptor)

def reduce(x, axes, keepdims, noop_with_empty_axes, reduce_type):
    reduce_layers = {
        0: torch.amax,
        1: torch.mean,
        2: torch.amin,
    }

    if reduce_type not in reduce_layers:
        raise ValueError(f"Invalid reduce_type: {reduce_type}. Supported: 0 (max), 1 (mean), 2 (min)")

    if axes is None and noop_with_empty_axes:
        return x

    reduction_fn = reduce_layers[reduce_type]

    if reduce_type in [0, 2]:
        result = reduction_fn(x, dim=axes, keepdim=keepdims)
    else:
        result = reduction_fn(x, dim=axes, keepdim=keepdims)

    return result

def inferShape(x_shape: List[int], axes: Optional[List[int]] = None, keepdims: bool = False, noop_with_empty_axes: bool = False) -> List[int]:
    if axes is None:
        if noop_with_empty_axes:
            return x_shape
        return [1] if not keepdims else [1] * len(x_shape)

    axes = [a if a >= 0 else len(x_shape) + a for a in axes]
    if keepdims:
        return [1 if i in axes else x_shape[i] for i in range(len(x_shape))]
    else:
        return [x_shape[i] for i in range(len(x_shape)) if i not in axes]

def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)

def test(lib, handle, torch_device, x_shape, axes, n_axes, keepdims, noop_with_empty_axes, reduce_type, tensor_dtype=torch.float16):
    reduce_name = "ReduceMin"
    print(f"Testing {reduce_name} on {torch_device} with x_shape: {x_shape}, axes: {axes}, dtype: {tensor_dtype}")

    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.rand(inferShape(x_shape, axes, keepdims, noop_with_empty_axes), dtype=tensor_dtype).to(torch_device)
    ans = reduce(x, axes, keepdims, noop_with_empty_axes, reduce_type)

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)

    descriptor = infiniopReduceMinDescriptor_t()

    check_error(lib.infiniopCreateReduceMinDescriptor(
        handle,
        ctypes.byref(descriptor),
        y_tensor.descriptor,
        x_tensor.descriptor,
        tuple_to_void_p(axes),
        n_axes,
        keepdims,
        noop_with_empty_axes,
    ))

    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    workspaceSize = ctypes.c_uint64(0)
    check_error(lib.infiniopGetReduceMinWorkspaceSize(descriptor, ctypes.byref(workspaceSize)))

    workspace = torch.zeros(int(workspaceSize.value), dtype=torch.uint8).to(torch_device)
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))

    check_error(lib.infiniopReduceMin(descriptor, workspace_ptr, workspaceSize, y_tensor.data, x_tensor.data, None))

    print("Output:", y)
    print("Expected:", ans)

    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyReduceMinDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, axes, n_axes, keepdims, noop_with_empty_axes, reduce_type in test_cases:
        test(lib, handle, "cpu", x_shape, axes, n_axes, keepdims, noop_with_empty_axes, reduce_type, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape, axes, n_axes, keepdims, noop_with_empty_axes, reduce_type, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        ((4, 4), [1], 1, True, False, 2),
        ((3, 3, 3), [0, 1], 2, False, True, 2),
        ((2, 4, 4, 4), [2], 1, True, False, 2),
    ]

    args = get_args()
    lib = open_lib()

    lib.infiniopCreateReduceMinDescriptor.restype = c_int32
    lib.infiniopCreateReduceMinDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopReduceMinDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_uint64,
        c_int,
        c_int,
    ]
    lib.infiniopReduceMin.restype = c_int32
    lib.infiniopReduceMin.argtypes = [
        infiniopReduceMinDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyReduceMinDescriptor.restype = c_int32
    lib.infiniopDestroyReduceMinDescriptor.argtypes = [infiniopReduceMinDescriptor_t]

    test_cpu(lib, test_cases)

    print("\033[92mReduceMin test passed!\033[0m")
