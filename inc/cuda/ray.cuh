#ifndef RAY_CUD_H
#define RAY_CUD_H
#include "vec.cuh"
#include <cuda_runtime.h>

class ray_gpu {
public:
  __device__ __host__ ray_gpu() : tm(0.0f) {}
  __device__ __host__ ray_gpu(const point3_gpu &origin,
                              const vec3_gpu &direction, float time = 0.0f)
      : orig(origin), dir(direction), tm(time) {}
  __device__ __host__ const point3_gpu &origin() const { return orig; }
  __device__ __host__ const vec3_gpu &direction() const { return dir; }
  __device__ __host__ float time() const { return tm; }
  __device__ __host__ point3_gpu at(float t) const { return orig + t * dir; }

private:
  point3_gpu orig;
  vec3_gpu dir;
  float tm;
};
#endif
