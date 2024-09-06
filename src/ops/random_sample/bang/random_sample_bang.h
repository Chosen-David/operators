#ifndef __BANG_RANDOM_SAMPLE_H__
#define __BANG_RANDOM_SAMPLE_H__

#include "../../../devices/bang/bang_handle.h"
#include "../../utils.h"
#include "operators.h"

struct RandomSampleBangDescriptor {
    Device device;
    int device_id;
    DT dtype;
    int voc;
};

typedef struct RandomSampleBangDescriptor *RandomSampleBangDescriptor_t;

infiniopStatus_t bangCreateRandomSampleDescriptor(BangHandle_t handle,
                                                  RandomSampleBangDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t probs);

infiniopStatus_t bangGetRandomSampleWorkspaceSize(RandomSampleBangDescriptor_t desc, unsigned long int *size);

infiniopStatus_t bangRandomSample(RandomSampleBangDescriptor_t desc,
                                  void *workspace,
                                  unsigned long int workspace_size,
                                  void *result,
                                  void *probs,
                                  float topp,
                                  int topk,
                                  float temperature,
                                  void *stream);

infiniopStatus_t bangDestroyRandomSampleDescriptor(RandomSampleBangDescriptor_t desc);


#endif
