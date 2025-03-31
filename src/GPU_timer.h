#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

struct GPU_Timer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GPU_Timer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GPU_Timer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

#endif  /* GPU_TIMER_H__ */
