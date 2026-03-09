#ifndef VEC3_CUD_H
#define VEC3_CUD_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
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

// Utility functions:
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

__device__ inline vec3_gpu
random_in_unit_sphere(curandState* local_rand_state) {
  vec3_gpu p;
  do {
    // Generate random vector in range [-1.0, 1.0] for x, y, z
    p = 2.0f * vec3_gpu(curand_uniform(local_rand_state),
                        curand_uniform(local_rand_state),
                        curand_uniform(local_rand_state)) -
        vec3_gpu(1, 1, 1);
  } while (p.length_squared() >= 1.0f);
  return p;
}

__device__ inline vec3_gpu random_unit_vector(curandState* local_rand_state) {
  return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ inline vec3_gpu random_in_unit_disk(curandState* local_rand_state) {
  vec3_gpu p;
  do {
    // Generate random vector in range [-1.0, 1.0] for x, y
    p = 2.0f * vec3_gpu(curand_uniform(local_rand_state),
                        curand_uniform(local_rand_state), 0.0f) -
        vec3_gpu(1.0f, 1.0f, 0.0f);
  } while (dot(p, p) >= 1.0f);
  return p;
}

__device__ inline vec3_gpu reflect(const vec3_gpu& v, const vec3_gpu& n) {
  return v - 2 * dot(v, n) * n;
}

__device__ inline vec3_gpu refract(const vec3_gpu& uv, const vec3_gpu& n,
                                   float etai_over_etat) {
  float cos_theta = fminf(dot(-uv, n), 1.0f);
  vec3_gpu r_out_perp = etai_over_etat * (uv + cos_theta * n);
  vec3_gpu r_out_parallel =
      -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
  return r_out_perp + r_out_parallel;
}

__device__ inline float reflectance(float cosine, float ref_idx) {
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

#endif
