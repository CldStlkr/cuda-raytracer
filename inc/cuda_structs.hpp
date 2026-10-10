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

struct HitRecordGPU {
  vec3_gpu p;
  vec3_gpu normal;
  float t;
  float v;
  float u;
  bool front_face;
  int material_id;
};

struct SphereGPU {
  vec3_gpu center;
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
  TextureGPU() {}
  union {
    struct {
      vec3_gpu color;
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
  vec3_gpu randvec[256];
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
  PrimitiveGPU() {}

  union {
    struct {
      vec3_gpu center;
      float radius;
    } sphere;

    struct {
      vec3_gpu center_start;
      vec3_gpu center_vec;
      float radius;
    } moving_sphere;

    struct {
      vec3_gpu Q, u, v, w;
      vec3_gpu normal;
      float D;
    } quad;
    struct {
      vec3_gpu center;
      float radius;
      float neg_inv_density;
    } volume_sphere;

    struct {
      vec3_gpu local_min;
      vec3_gpu local_max;
      vec3_gpu offset;
      float sin_theta;
      float cos_theta;
      float neg_inv_density;
    } volume_box;
    struct {
      vec3_gpu Q_start, Q_vec;
      vec3_gpu u, v, w;
      vec3_gpu normal;
      float D_start, D_vec;
    } moving_quad;
  };

  // AABB for BHV intersection fast-path
  vec3_gpu aabb_min;
  vec3_gpu aabb_max;

  int material_id;
};

struct LinearBVHNode {
  vec3_gpu aabb_min;
  vec3_gpu aabb_max;

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
  float4* frame_buffer;
  int width;
  int height;
  int samples_per_pixel;
  int max_depth;
  vec3_gpu background;

  // Camera Settings
  vec3_gpu lookfrom;
  vec3_gpu lookat;
  vec3_gpu vup;
  float vfov;
  // Shutter interval must match the host reference renderer's interval
  // exactly, or a differential validation measures motion blur instead of
  // precision. the host camera draws random_double() in [0,1), and
  // sphere(c1,c2, ...) interpolates center over t in [0,1].
  float time0 = 0.0f;
  float time1 = 1.0f;
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
  int* pixel_index;
  int* sample_index; // global sample index — used as RNG seed component
  int* depth;
  bool* alive;
};

struct HitResultSOA {
  // HitRecordGPU components (decomposed for SoA)
  vec3_gpu* hit_p;
  vec3_gpu* hit_normal;
  float* hit_t;
  float* hit_u;
  float* hit_v;
  bool* hit_front_face;
  int* hit_material_id;

  bool* hit_anything;
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
