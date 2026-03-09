#ifndef SPHERE_CUD_H
#define SPHERE_CUD_H
#include "ray.cuh"
#include "vec.cuh"
#include <cuda_runtime.h>

enum MaterialType {
  MAT_DIFFUSE,
  MAT_METAL,
  MAT_DIELECTRIC,
};

class sphere_gpu {
public:
  MaterialType mat_type;
  vec3_gpu albedo; // Color of material
  float fuzz;      // How perfectly reflective it is (0.0 - 1.0)
  float ir;

  __host__ __device__ sphere_gpu() {}

  // Diffuse Constructor
  __host__ __device__ sphere_gpu(const point3_gpu& c, float r, vec3_gpu color)
      : mat_type{MAT_DIFFUSE}, albedo{color}, fuzz{0.0f}, ir{0.0f}, center{c},
        radius{fmaxf(0.0f, r)} {}
  // Metal Constructor
  __host__ __device__ sphere_gpu(const point3_gpu& c, float r, vec3_gpu color,
                                 float f)
      : mat_type{MAT_METAL}, albedo{color}, fuzz{f < 1 ? f : 1}, ir{0.0f},
        center{c}, radius{fmaxf(0.0f, r)} {}

  // Dielectric Constructor
  __host__ __device__ sphere_gpu(const point3_gpu& c, float r,
                                 float index_of_refraction)
      : mat_type{MAT_DIELECTRIC}, albedo{vec3_gpu(1.0f, 1.0f, 1.0f)},
        fuzz{0.0f}, ir{index_of_refraction}, center{c}, radius{fmaxf(0.0f, r)} {
  }

  __host__ __device__ bool hit(const ray_gpu& r, float t_min, float t_max,
                               float& t) const {
    vec3_gpu oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float half_b = dot(oc, r.direction());
    float c_coeff = dot(oc, oc) - radius * radius; // Fixed variable name
    float discriminant = half_b * half_b - a * c_coeff;

    if (discriminant < 0) return false;
    float sqrtd = sqrtf(discriminant);

    // Find nearest root in [t_min, t_max]
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
      root = (-half_b + sqrtd) / a;
      if (root < t_min || root > t_max) return false;
    }
    t = root;
    return true;
  }

  __host__ __device__ point3_gpu get_center() const { return center; }
  __host__ __device__ float get_radius() const { return radius; }

private:
  point3_gpu center;
  float radius;
};
#endif
