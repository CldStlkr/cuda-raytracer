#ifndef VEC3_CUD_H
#define VEC3_CUD_H
#include <cuda_runtime.h>
#include <math.h>

class vec3_gpu {
public:
  float e[3];
  __host__ __device__ vec3_gpu() : e{0.0f, 0.0f, 0.0f} {}
  __host__ __device__ vec3_gpu(float e0, float e1, float e2) : e{e0, e1, e2} {}
  __host__ __device__ float x() const { return e[0]; }
  __host__ __device__ float y() const { return e[1]; }
  __host__ __device__ float z() const { return e[2]; }
  __host__ __device__ float operator[](int i) const { return e[i]; }
  __host__ __device__ float& operator[](int i) { return e[i]; }
  __host__ __device__ vec3_gpu operator-() const {
    return vec3_gpu(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ vec3_gpu& operator+=(const vec3_gpu& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }
  __host__ __device__ vec3_gpu& operator*=(float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }
  __host__ __device__ vec3_gpu& operator/=(float t) {
    return *this *= 1.0f / t;
  }
  __host__ __device__ float length() const { return sqrtf(length_squared()); }
  __host__ __device__ float length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }
};

// Aliases
using point3_gpu = vec3_gpu;
using color_gpu = vec3_gpu;

// Utility functions
__host__ __device__ inline vec3_gpu operator+(const vec3_gpu& u,
                                              const vec3_gpu& v) {
  return vec3_gpu(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
__host__ __device__ inline vec3_gpu operator-(const vec3_gpu& u,
                                              const vec3_gpu& v) {
  return vec3_gpu(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
__host__ __device__ inline vec3_gpu operator*(const vec3_gpu& u,
                                              const vec3_gpu& v) {
  return vec3_gpu(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}
__host__ __device__ inline vec3_gpu operator*(float t, const vec3_gpu& v) {
  return vec3_gpu(t * v.e[0], t * v.e[1], t * v.e[2]);
}
__host__ __device__ inline vec3_gpu operator*(const vec3_gpu& v, float t) {
  return t * v;
}
__host__ __device__ inline vec3_gpu operator/(const vec3_gpu& v, float t) {
  return (1.0f / t) * v;
}
__host__ __device__ inline float dot(const vec3_gpu& u, const vec3_gpu& v) {
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}
__host__ __device__ inline vec3_gpu cross(const vec3_gpu& u,
                                          const vec3_gpu& v) {
  return vec3_gpu(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                  u.e[2] * v.e[0] - u.e[0] * v.e[2],
                  u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}
__host__ __device__ inline vec3_gpu unit_vector(const vec3_gpu& v) {
  return v / v.length();
}
__host__ __device__ inline vec3_gpu normalize(const vec3_gpu& v) {
  return unit_vector(v);
}
#endif
