﻿#include "rms_norm.h"
#include "../../utils.h"
#include "cnrt.h"
#include "cnnl.h"
#include "cnnl_extra.h"

void rms_norm_cambricon_mlu_f16(MutTensor y, ConstTensor x, ConstTensor w, float epsilon, void* stream) {
    ASSERT_EQ(y.layout.ndim, 2);
    ASSERT_EQ(x.layout.ndim, 2);
    ASSERT_EQ(w.layout.ndim, 1);

    auto n = y.layout.shape[0],
         d = y.layout.shape[1];

    ASSERT_EQ(x.layout.shape[0], n);
    ASSERT_EQ(x.layout.shape[1], d);
    ASSERT_EQ(w.layout.shape[0], d);
    
    cnnlTensorDescriptor_t yDesc, xDesc, wDesc;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&xDesc);
    cnnlCreateTensorDescriptor(&wDesc);

    setCnnlTensor(yDesc, y.layout);
    setCnnlTensor(xDesc, x.layout);
    setCnnlTensor(wDesc, w.layout);

    cnnlFuseNormDescriptor_t opDesc;
    cnnlCreateFuseNormDescriptor(&opDesc);
    cnnlSetFuseNormDescriptor(opDesc, epsilon, 1.0, true,
                              false, false, false, false,
                              CNNL_DTYPE_HALF, CNNL_TRANSFORMER_RMSNORM);

    auto handle = getCnnlHandle(stream);

    size_t wsSize;
    cnnlGetFuseNormWorkspaceSize(handle, opDesc, xDesc, &wsSize);

    void *workspace;
    cnrtMalloc(&workspace, wsSize);

    cnnlFuseNorm(handle, opDesc, xDesc, x.data,
                 wDesc, w.data, nullptr, nullptr,
                 nullptr, nullptr, nullptr, nullptr,
                 workspace, wsSize, yDesc, y.data, nullptr, nullptr);

    cnrtFree(workspace);
    cnnlDestroyFuseNormDescriptor(opDesc);
    cnnlDestroyTensorDescriptor(xDesc);
    cnnlDestroyTensorDescriptor(yDesc);
    cnnlDestroyTensorDescriptor(wDesc);
}
