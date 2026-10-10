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
__host__ __device__ inline uint32_t pcg_hash(uint32_t x) {
  x = x * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}

// Register-only RNG: state lives entirely in registers, zero global memory I/O.
// Seeded from (pixel, sample, bounce, stream) where stream distinguishes kernel
// stages: 0 = generate_rays, 1 = intersect_bvh, 2 = shade_kernel.
//
// Host-callable so the float64 reference renderer can consume the identical
// draw sequence. The seed is a pure function of the coordinate tuple, so
// evaluation order doesn't matter — a wavefront GPU traversal and a recursive
// host traversal still see the same stream.
struct RNG {
  uint64_t state;

  __host__ __device__ RNG(uint32_t pixel, uint32_t sample, uint32_t bounce, uint32_t stream) {
    uint32_t s0 = pcg_hash(pixel ^ pcg_hash(sample));
    uint32_t s1 = pcg_hash(bounce ^ pcg_hash(stream));
    state = (uint64_t(s1) << 32) | uint64_t(s0);
    next_u32(); // warm-up: mix seed into full state
  }

  __host__ __device__ uint32_t next_u32() {
    uint64_t old = state;
    state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = uint32_t(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
  }

  __host__ __device__ float next_f() { return (next_u32() >> 8) * (1.0f / 16777216.0f); }

  // Same 24 bits as next_f() and the same single draw, widened to double.
  // Not 53 bits: a second draw would desync the host and device streams, and
  // would change the integrand rather than just its precision.
  __host__ __device__ double next_d() { return (next_u32() >> 8) * (1.0 / 16777216.0); }
};

// sincosf is device-only. Keep the accurate form, not __sincosf, whose reduced
// precision would show up in a precision comparison.
__host__ __device__ inline void sin_cos_f(float a, float& s, float& c) {
#ifdef __CUDA_ARCH__
  sincosf(a, &s, &c);
#else
  s = sinf(a);
  c = cosf(a);
#endif
}

// Uniform on the unit sphere (Archimedes): z uniform on [-1,1], azimuth uniform.
// Exactly 2 draws, no branch. Replaces a rejection loop with ~52% acceptance:
// a fixed draw count is required to keep the host and device streams aligned,
// since the f32 and f64 acceptance tests can disagree near the boundary. Also
// removes the warp looping until its last lane accepts, and the normalize by a
// length with no lower bound.
__host__ __device__ inline vec3_gpu random_unit_vector(RNG& rng) {
  float z = 1.0f - 2.0f * rng.next_f();
  float phi = 6.28318530717958648f * rng.next_f();
  // (1-z)*(1+z), not 1-z*z: 1-z is exact for z in [0.5,2] (Sterbenz), so this
  // avoids the cancellation as |z|->1. The 1-z*z form differs host-vs-device by
  // up to 511 ULP, since nvcc contracts it to an FMA and the host does not.
  float r = sqrtf(fmaxf(0.0f, (1.0f - z) * (1.0f + z))); // |z| <= 1; fmaxf guards rounding only
  float s, c;
  sin_cos_f(phi, s, c);
  return vec3_gpu(r * c, r * s, z);
}

// Uniform inside the unit sphere: uniform direction, r = cbrt(u) since volume
// grows as r^3. Exactly 3 draws.
__host__ __device__ inline vec3_gpu random_in_unit_sphere(RNG& rng) {
  vec3_gpu dir = random_unit_vector(rng);
  float r = cbrtf(rng.next_f());
  return r * dir;
}

// Uniform on the unit disk: r = sqrt(u) since area grows as r^2. Exactly 2 draws.
__host__ __device__ inline vec3_gpu random_in_unit_disk(RNG& rng) {
  float r = sqrtf(rng.next_f());
  float theta = 6.28318530717958648f * rng.next_f();
  float s, c;
  sin_cos_f(theta, s, c);
  return vec3_gpu(r * c, r * s, 0.0f);
}

__host__ __device__ inline vec3_gpu reflect(const vec3_gpu& v, const vec3_gpu& n) {
  return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline vec3_gpu refract(const vec3_gpu& uv, const vec3_gpu& n, float etai_over_etat) {
  float cos_theta = fminf(dot(-uv, n), 1.0f);
  vec3_gpu r_out_perp = etai_over_etat * (uv + cos_theta * n);
  // |r_out_perp| == etai_over_etat * sin(theta), so the parallel term is
  // sqrt(1 - w^2) with w = eta*sin(theta). Factoring as (1-w)(1+w) drops a
  // length_squared() and reduces worst-case relative error near the critical
  // angle from 11.3% to 7.4% (measured) -- it does not fix it: w -> 1 there,
  // so cos_t is near zero and the result is intrinsically ill-conditioned in
  // f32. Judge this on absolute error, not relative or ULP.
  // w > 1 is total internal reflection, which the caller screens for;
  // clamping to 0 there is the grazing limit.
  float sin_theta = sqrtf(fmaxf(0.0f, (1.0f - cos_theta) * (1.0f + cos_theta)));
  float w = etai_over_etat * sin_theta;
  float cos_t = sqrtf(fmaxf(0.0f, (1.0f - w) * (1.0f + w)));
  return r_out_perp - cos_t * n;
}

__host__ __device__ inline float reflectance(float cosine, float ref_idx) {
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  float x = 1.0f - cosine;
  float x2 = x * x;
  return r0 + (1.0f - r0) * (x2 * x2 * x); // x^5 by multiplication; powf is a transcendental
}

#endif
