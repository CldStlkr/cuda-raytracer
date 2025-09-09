#pragma once
#include <cuda_runtime.h>

struct Vec3f {
  float x, y, z;
};

struct RayGPU {
  Vec3f origin;
  Vec3f direction;
};

struct SphereGPU {
  Vec3f center;
  float radius;
  int material_id; // index into material array
};

enum MaterialType {
  LAMBERTIAN = 0,
  METAL = 1,
  DIELECTRIC = 2,
};

struct MaterialGPU {
  int type; // Uses MaterialType enum
  Vec3f albedo;
  float fuzz;    // Only relevant for METAL
  float ref_idx; // Only relevant for DIELECTRIC
};
