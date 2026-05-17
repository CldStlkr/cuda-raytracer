#pragma once
#include "cuda/ray.cuh"
#include "cuda/vec.cuh"
#include <cstdint>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "bvh.hpp"
#include "hittable.hpp"
#include "quad.hpp"
#include "sphere.hpp"

namespace cuda {
using cuda::std::span;
}

struct Vec3f {
  float x, y, z;
};

inline Vec3f to_vec3f(const vec3& v) {
  return Vec3f{static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z())};
}

struct RayGPU {
  Vec3f origin;
  Vec3f direction;
};

struct HitRecordGPU {
  Vec3f p;
  Vec3f normal;
  float t;
  float v;
  float u;
  bool front_face;
  int material_id;
};

struct SphereGPU {
  Vec3f center;
  float radius;
  int material_id; // index into material array
};

enum class MaterialType : uint8_t {
  LAMBERTIAN = 0,
  METAL = 1,
  DIELECTRIC = 2,
  DIFFUSE_LIGHT = 3,
  ISOTROPIC = 4,
};

enum class TextureType : uint8_t {
  SOLID = 0,
  CHECKER = 1,
  IMAGE = 2,
  NOISE = 3,
};

struct TextureGPU {
  TextureType type;
  union {
    struct {
      Vec3f color;
    } solid;
    struct {
      float inv_scale;
      int even_tex_idx;
      int odd_tex_idx;
    } checker;
    struct {
      int width;
      int height;
      int bytes_per_scanline;
      int offset;
    } image;
    struct {
      float scale;
      int perlin_data_idx;
    } noise;
  };
};

struct PerlinDataGPU {
  Vec3f randvec[256];
  int perm_x[256];
  int perm_y[256];
  int perm_z[256];
};

struct MaterialGPU {
  MaterialType type;
  int albedo_tex_id; // index into texture buffer
  float fuzz;
  float ref_idx;
};

enum class PrimitiveType : int {
  SPHERE = 0,
  QUAD = 1,
  MOVING_SPHERE = 2,
  VOLUME_SPHERE = 3,
  VOLUME_BOX = 4,
  MOVING_QUAD = 5,
};

struct PrimitiveGPU {
  PrimitiveType type;

  union {
    struct {
      Vec3f center;
      float radius;
    } sphere;

    struct {
      Vec3f center_start;
      Vec3f center_vec;
      float radius;
    } moving_sphere;

    struct {
      Vec3f Q, u, v, w;
      Vec3f normal;
      float D;
    } quad;
    struct {
      Vec3f center;
      float radius;
      float neg_inv_density;
    } volume_sphere;

    struct {
      Vec3f local_min;
      Vec3f local_max;
      Vec3f offset;
      float sin_theta;
      float cos_theta;
      float neg_inv_density;
    } volume_box;
    struct {
      Vec3f Q_start, Q_vec;
      Vec3f u, v, w;
      Vec3f normal;
      float D_start, D_vec;
    } moving_quad;
  };

  // AABB for BHV intersection fast-path
  Vec3f aabb_min;
  Vec3f aabb_max;

  int material_id;
};

struct LinearBVHNode {
  Vec3f aabb_min;
  Vec3f aabb_max;

  union {
    int primitive_offset;    // Leaf node: index into the PrimitiveGPU array
    int second_child_offset; // Interior node: index of the right child in the
                             // LinearBVHNode array
  };

  uint16_t n_primitives; // 0 if interior node, > 0 if leaf
  uint8_t axis;
  uint8_t pad;
};

struct RenderConfig {
  vec3_gpu* frame_buffer;
  int width;
  int height;
  int samples_per_pixel;
  int max_depth;
  Vec3f background;

  // Camera Settings
  Vec3f lookfrom;
  Vec3f lookat;
  Vec3f vup;
  float vfov;
  float defocus_angle;
  float focus_dist;
};

// SoA (Structure-of-Arrays) layout for coalesced GPU memory access.
// Each field is a separate contiguous array of size 'total_rays'.
// When threads 0-31 in a warp read ray_origin[0..31], that's a single
// coalesced 128-byte cache line transaction instead of 32 scattered reads.
struct PathStateSOA {
  // Ray components (decomposed from ray_gpu for SoA)
  vec3_gpu* ray_origin;
  vec3_gpu* ray_dir;
  float* ray_time;

  // Path state
  vec3_gpu* attenuation;
  vec3_gpu* color;
  int* pixel_index;
  int* depth;
  bool* alive;
};

struct HitResultSOA {
  // HitRecordGPU components (decomposed for SoA)
  Vec3f* hit_p;
  Vec3f* hit_normal;
  float* hit_t;
  float* hit_u;
  float* hit_v;
  bool* hit_front_face;
  int* hit_material_id;

  bool* hit_anything;
};

#define NUM_MATERIAL_TYPES 5

// Per-material queues for warp-aggregated atomic dispatch.
// Replaces thrust::sort_by_key + thrust::partition with O(1) per-thread
// queue pushes using __ballot_sync and __shfl_sync.
struct MaterialQueues {
  int* queues[NUM_MATERIAL_TYPES]; // per-material index arrays (preallocated to
                                   // total_rays)
  int* counts;                     // device array of NUM_MATERIAL_TYPES ints (atomic counters)
  int* next_active;                // queue for next bounce's active indices
  int* next_count;                 // device int (atomic counter for next_active)
};

int get_or_add_texture(std::shared_ptr<texture> tex_ptr, std::vector<TextureGPU>& linear_textures,
                       std::vector<PerlinDataGPU>& linear_perlin, std::vector<unsigned char>& image_buffer,
                       std::unordered_map<texture*, int>& tex_map);

int get_or_add_material(std::shared_ptr<material> mat_ptr, std::vector<MaterialGPU>& linear_materials,
                        std::vector<TextureGPU>& linear_textures, std::vector<PerlinDataGPU>& linear_perlin,
                        std::vector<unsigned char>& image_buffer, std::unordered_map<material*, int>& mat_map,
                        std::unordered_map<texture*, int>& tex_map);

int flatten_hittable(std::shared_ptr<hittable> node, std::vector<LinearBVHNode>& linear_nodes,
                     std::vector<PrimitiveGPU>& linear_primitives, std::vector<MaterialGPU>& linear_materials,
                     std::vector<TextureGPU>& linear_textures, std::vector<PerlinDataGPU>& linear_perlin,
                     std::vector<unsigned char>& image_buffer, std::unordered_map<material*, int>& mat_map,
                     std::unordered_map<texture*, int>& tex_map);
