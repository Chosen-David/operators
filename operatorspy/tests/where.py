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
from enum import Enum, auto
import torch

class Inplace(Enum):
    OUT_OF_PLACE = auto()  # 生成新的 output
    INPLACE_X = auto()     # 直接使用 x 作为 output
    INPLACE_Y = auto()     # 直接使用 y 作为 output

class WhereDescriptor(Structure):
    _fields_=[("device",c_int32)]

infiniopWhereDescriptor_t=POINTER(WhereDescriptor)

def where(condition, x, y):
    return torch.where(condition, x, y)

def test(
    lib,
    handle,
    torch_device,
    output_shape, 
    x_shape, 
    y_shape,
    condition_shape,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Where on {torch_device} with output_shape:{output_shape} x_shape:{x_shape} y_shape:{y_shape} condition_shape:{condition_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    if x_shape != y_shape and inplace != Inplace.OUT_OF_PLACE:
        print("Unsupported test: broadcasting does not support in-place")
        return
    
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.rand(y_shape, dtype=tensor_dtype).to(torch_device)
    condition = torch.randint(0, 2, condition_shape, dtype=torch.uint8).to(torch_device)
    output = torch.rand(output_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else (x if inplace == Inplace.INPLACE_X else y)

    ans=where(condition.to(torch.bool), x, y)
   

    x_tensor=to_tensor(x,lib)
    y_tensor=to_tensor(y,lib)
    condition_tensor=to_tensor(condition,lib)
    output_tensor=to_tensor(output,lib) if inplace==Inplace.OUT_OF_PLACE else(x_tensor if inplace==Inplace.INPLACE_X else y_tensor )
    descriptor = infiniopWhereDescriptor_t()

    check_error(
        lib.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            condition_tensor.descriptor,
            x_tensor.descriptor,
            y_tensor.descriptor,
            output_tensor.descriptor,
        )
    )

    
    x_tensor.descriptor.contents.invalidate()
    condition_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    output_tensor.descriptor.contents.invalidate()
   

    check_error(
        lib.infiniopWhere(descriptor,condition_tensor.data,x_tensor.data,y_tensor.data,output_tensor.data,None)
    )

    assert torch.allclose(output,ans,atol=0,rtol=0)
    check_error(lib.infiniopDestroyWhereDescriptor(descriptor))

def test_cpu(lib,test_cases):
    device=DeviceEnum.DEVICE_CPU
    handle=create_handle(lib,device)
    for condition_shape,x_shape,y_shape,output_shape,inplace in test_cases:
        test(lib,handle,"cpu",condition_shape,x_shape,y_shape,output_shape,tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu",condition_shape,x_shape,y_shape,output_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib,handle)

if __name__ == "__main__":

    test_cases = [
        # (condition_shape, x_shape, y_shape, output_shape, inplace)
        ((3, 3), (3, 3), (3, 3), (3, 3), Inplace.OUT_OF_PLACE),  
        ((1, 3), (1, 3), (1, 3), (1, 3), Inplace.OUT_OF_PLACE),  
        ((), (), (), (), Inplace.OUT_OF_PLACE),  
        ((2, 20, 3), (2, 20, 3), (2, 20, 3), (2, 20, 3), Inplace.OUT_OF_PLACE),  
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.OUT_OF_PLACE),  
        ((32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),  
        ((2, 4, 3), (2, 4, 3), (2, 4, 3), (2, 4, 3), Inplace.OUT_OF_PLACE),  
        ((2, 3, 4, 5), (2, 3, 4, 5), (2, 3, 4, 5), (2, 3, 4, 5), Inplace.OUT_OF_PLACE),  
        ((3, 2, 4, 5), (3, 2, 4, 5), (3, 2, 4, 5), (3, 2, 4, 5), Inplace.OUT_OF_PLACE),  
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_X),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_Y),
        ((2, 4, 3), (2, 4, 3), (4, 3), (2, 4, 3), Inplace.OUT_OF_PLACE),  
        ((2, 3, 4, 5), (2, 3, 4, 5), (5,), (2, 3, 4, 5), Inplace.OUT_OF_PLACE),  
        ((3, 2, 4, 5), (4, 5), (3, 2, 1, 1), (3, 2, 4, 5), Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()

    # 设定 where 算子函数签名
    lib.infiniopCreateWhereDescriptor.restype = c_int32
    lib.infiniopCreateWhereDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopWhereDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopWhere.restype = c_int32
    lib.infiniopWhere.argtypes = [
        infiniopWhereDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyWhereDescriptor.restype = c_int32
    lib.infiniopDestroyWhereDescriptor.argtypes = [
        infiniopWhereDescriptor_t,
    ]

    # 运行测试
    if args.cpu:
        test_cpu(lib, test_cases)

    if not (args.cpu):
        test_cpu(lib, test_cases)

    print("\033[92mTest passed!\033[0m")

