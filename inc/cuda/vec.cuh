#ifndef VEC3_CUD_H
#define VEC3_CUD_H

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

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
  __host__ __device__ vec3_gpu operator-() const { return vec3_gpu(-e[0], -e[1], -e[2]); }
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
  __host__ __device__ vec3_gpu& operator/=(float t) { return *this *= 1.0f / t; }
  __host__ __device__ float length() const { return sqrtf(length_squared()); }
  __host__ __device__ float length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
};

// Aliases
using point3_gpu = vec3_gpu;
using color_gpu = vec3_gpu;

// Utility functions:
__host__ __device__ inline vec3_gpu operator+(const vec3_gpu& u, const vec3_gpu& v) {
  return vec3_gpu(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
__host__ __device__ inline vec3_gpu operator+(const vec3_gpu& v, float t) {
  return vec3_gpu(v.e[0] + t, v.e[1] + t, v.e[2] + t);
}
__host__ __device__ inline vec3_gpu operator+(float t, const vec3_gpu& v) { return v + t; }
__host__ __device__ inline vec3_gpu operator-(const vec3_gpu& u, const vec3_gpu& v) {
  return vec3_gpu(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
__host__ __device__ inline vec3_gpu operator*(const vec3_gpu& u, const vec3_gpu& v) {
  return vec3_gpu(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}
__host__ __device__ inline vec3_gpu operator*(float t, const vec3_gpu& v) {
  return vec3_gpu(t * v.e[0], t * v.e[1], t * v.e[2]);
}
__host__ __device__ inline vec3_gpu operator*(const vec3_gpu& v, float t) { return t * v; }
__host__ __device__ inline vec3_gpu operator/(const vec3_gpu& v, float t) { return (1.0f / t) * v; }
__host__ __device__ inline vec3_gpu operator/(const vec3_gpu& u, const vec3_gpu& v) {
  return vec3_gpu(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]);
}
__host__ __device__ inline float dot(const vec3_gpu& u, const vec3_gpu& v) {
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}
__host__ __device__ inline vec3_gpu cross(const vec3_gpu& u, const vec3_gpu& v) {
  return vec3_gpu(u.e[1] * v.e[2] - u.e[2] * v.e[1], u.e[2] * v.e[0] - u.e[0] * v.e[2],
                  u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}
__host__ __device__ inline vec3_gpu unit_vector(const vec3_gpu& v) { return v / v.length(); }
__host__ __device__ inline vec3_gpu normalize(const vec3_gpu& v) { return unit_vector(v); }

// PCG hash for seeding — good avalanche, one multiply+xorshift pair.
__device__ inline uint32_t pcg_hash(uint32_t x) {
  x = x * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}

// Register-only RNG: state lives entirely in registers, zero global memory I/O.
// Seeded from (pixel, sample, bounce, stream) where stream distinguishes kernel
// stages: 0 = generate_rays, 1 = intersect_bvh, 2 = shade_kernel.
struct RNG {
  uint64_t state;

  __device__ RNG(uint32_t pixel, uint32_t sample, uint32_t bounce, uint32_t stream) {
    uint32_t s0 = pcg_hash(pixel ^ pcg_hash(sample));
    uint32_t s1 = pcg_hash(bounce ^ pcg_hash(stream));
    state = (uint64_t(s1) << 32) | uint64_t(s0);
    next_u32(); // warm-up: mix seed into full state
  }

  __device__ uint32_t next_u32() {
    uint64_t old = state;
    state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = uint32_t(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
  }

  __device__ float next_f() { return (next_u32() >> 8) * (1.0f / 16777216.0f); }
};

__device__ inline vec3_gpu random_in_unit_sphere(RNG& rng) {
  vec3_gpu p;
  do {
    p = 2.0f * vec3_gpu(rng.next_f(), rng.next_f(), rng.next_f()) - vec3_gpu(1, 1, 1);
  } while (p.length_squared() >= 1.0f);
  return p;
}

__device__ inline vec3_gpu random_unit_vector(RNG& rng) {
  return unit_vector(random_in_unit_sphere(rng));
}

__device__ inline vec3_gpu random_in_unit_disk(RNG& rng) {
  vec3_gpu p;
  do {
    p = 2.0f * vec3_gpu(rng.next_f(), rng.next_f(), 0.0f) - vec3_gpu(1.0f, 1.0f, 0.0f);
  } while (dot(p, p) >= 1.0f);
  return p;
}

__device__ inline vec3_gpu reflect(const vec3_gpu& v, const vec3_gpu& n) { return v - 2 * dot(v, n) * n; }

__device__ inline vec3_gpu refract(const vec3_gpu& uv, const vec3_gpu& n, float etai_over_etat) {
  float cos_theta = fminf(dot(-uv, n), 1.0f);
  vec3_gpu r_out_perp = etai_over_etat * (uv + cos_theta * n);
  vec3_gpu r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
  return r_out_perp + r_out_parallel;
}

__device__ inline float reflectance(float cosine, float ref_idx) {
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

#endif
