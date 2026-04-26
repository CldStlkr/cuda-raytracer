#pragma once
#include "cuda/vec.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "bvh.hpp"
#include "hittable.hpp"
#include "quad.hpp"
#include "sphere.hpp"

struct Vec3f {
  float x, y, z;
};

inline Vec3f to_vec3f(const vec3 &v) {
  return Vec3f{static_cast<float>(v.x()), static_cast<float>(v.y()),
               static_cast<float>(v.z())};
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

enum class MaterialType : int {
  LAMBERTIAN = 0,
  METAL = 1,
  DIELECTRIC = 2,
  DIFFUSE_LIGHT = 3,
};

enum class TextureType : int {
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
  vec3_gpu *frame_buffer;
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

struct BVHBuffer {
  const LinearBVHNode *data;
  size_t count;
};

struct PrimitiveBuffer {
  const PrimitiveGPU *data;
  size_t count;
};

struct MaterialBuffer {
  const MaterialGPU *data;
  size_t count;
};

struct TextureBuffer {
  const TextureGPU *data;
  size_t count;
};

struct PerlinBuffer {
  const PerlinDataGPU *data;
  size_t count;
};

struct ImageArrayBuffer {
  const unsigned char *data;
  size_t count_bytes;
};

int get_or_add_texture(std::shared_ptr<texture> tex_ptr,
                       std::vector<TextureGPU> &linear_textures,
                       std::vector<PerlinDataGPU> &linear_perlin,
                       std::vector<unsigned char> &image_buffer,
                       std::unordered_map<texture *, int> &tex_map);

int get_or_add_material(std::shared_ptr<material> mat_ptr,
                        std::vector<MaterialGPU> &linear_materials,
                        std::vector<TextureGPU> &linear_textures,
                        std::vector<PerlinDataGPU> &linear_perlin,
                        std::vector<unsigned char> &image_buffer,
                        std::unordered_map<material *, int> &mat_map,
                        std::unordered_map<texture *, int> &tex_map);

int flatten_hittable(std::shared_ptr<hittable> node,
                     std::vector<LinearBVHNode> &linear_nodes,
                     std::vector<PrimitiveGPU> &linear_primitives,
                     std::vector<MaterialGPU> &linear_materials,
                     std::vector<TextureGPU> &linear_textures,
                     std::vector<PerlinDataGPU> &linear_perlin,
                     std::vector<unsigned char> &image_buffer,
                     std::unordered_map<material *, int> &mat_map,
                     std::unordered_map<texture *, int> &tex_map);
