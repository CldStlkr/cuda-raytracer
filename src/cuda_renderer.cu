#include "cuda/bvh_kernel.cuh"
#include "cuda/camera.cuh"
#include "cuda/ray.cuh"
#include "cuda/vec.cuh"

#include "cuda_structs.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <stdio.h>

__host__ __device__ inline vec3_gpu to_gpu(const Vec3f& v) { return vec3_gpu(v.x, v.y, v.z); }
__host__ __device__ inline Vec3f to_vec3f(const vec3_gpu& v) { return {v.x(), v.y(), v.z()}; }

__global__ __launch_bounds__(256, 2) void generate_rays(
    PathStateSOA paths, camera_gpu cam,
    curandState* rand_state,
    int width, int height, int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_rays = width * height * batch_size;
  if (idx >= total_rays) return;

  int pixel_index = idx / batch_size;
  int i = pixel_index % width;
  int j = pixel_index / width;

  curandState local_rand = rand_state[idx];
  float u = float(i) + (curand_uniform(&local_rand) - 0.5f);
  float v = float(j) + (curand_uniform(&local_rand) - 0.5f);

  ray_gpu r = cam.get_ray(u, v, &local_rand);
  paths.ray_origin[idx] = r.origin();
  paths.ray_dir[idx] = r.direction();
  paths.ray_time[idx] = r.time();
  paths.attenuation[idx] = vec3_gpu(1.0f, 1.0f, 1.0f);
  paths.color[idx] = vec3_gpu(0.0f, 0.0f, 0.0f);
  paths.pixel_index[idx] = pixel_index;
  paths.depth[idx] = 0;
  paths.alive[idx] = true;

  rand_state[idx] = local_rand;
}

__global__ void render_init(int total_rays, curandState* rand_state) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= total_rays) return;
  curand_init(1984, idx, 0, &rand_state[idx]);
}

__global__ void intersect_bvh(PathStateSOA paths, HitResultSOA hits, int* active_indices, int num_active,
                              const LinearBVHNode* bvh_nodes, const PrimitiveGPU* primitives, curandState* rand_state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_active) return;
  int path_idx = active_indices[idx];
  
  ray_gpu r(paths.ray_origin[path_idx], paths.ray_dir[path_idx], paths.ray_time[path_idx]);
  HitRecordGPU rec;
  bool hit = hit_linear_bvh(bvh_nodes, primitives, r, 0.001f, 1e20f, rec, &rand_state[path_idx]);
  
  hits.hit_anything[idx] = hit;
  if (hit) {
    hits.hit_p[idx] = rec.p;
    hits.hit_normal[idx] = rec.normal;
    hits.hit_t[idx] = rec.t;
    hits.hit_u[idx] = rec.u;
    hits.hit_v[idx] = rec.v;
    hits.hit_front_face[idx] = rec.front_face;
    hits.hit_material_id[idx] = rec.material_id;
  }
}

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(err));                                         \
      return;                                                                  \
    }                                                                          \
  } while (0)

__device__ float perlin_interp(const vec3_gpu c[2][2][2], float u, float v,
                               float w) {
  float uu = u * u * (3.0f - 2.0f * u);
  float vv = v * v * (3.0f - 2.0f * v);
  float ww = w * w * (3.0f - 2.0f * w);
  float accum = 0.0f;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++) {
        vec3_gpu weight_v(u - i, v - j, w - k);
        accum += (i * uu + (1.0f - i) * (1.0f - uu)) *
                 (j * vv + (1.0f - j) * (1.0f - vv)) *
                 (k * ww + (1.0f - k) * (1.0f - ww)) *
                 dot(c[i][j][k], weight_v);
      }
  return accum;
}

__device__ float perlin_noise(const point3_gpu& p, const PerlinDataGPU& pdata) {
  auto u = p.x() - floorf(p.x());
  auto v = p.y() - floorf(p.y());
  auto w = p.z() - floorf(p.z());
  int i = int(floorf(p.x())), j = int(floorf(p.y())), k = int(floorf(p.z()));
  vec3_gpu c[2][2][2];
  for (int di = 0; di < 2; di++)
    for (int dj = 0; dj < 2; dj++)
      for (int dk = 0; dk < 2; dk++) {
        int idx = pdata.perm_x[(i + di) & 255] ^ pdata.perm_y[(j + dj) & 255] ^
                  pdata.perm_z[(k + dk) & 255];
        c[di][dj][dk] = to_gpu(pdata.randvec[idx]);
      }
  return perlin_interp(c, u, v, w);
}

__device__ float perlin_turb(const point3_gpu& p, const PerlinDataGPU& pdata,
                             int depth) {
  float accum = 0.0f;
  point3_gpu temp_p = p;
  float weight = 1.0f;
  for (int i = 0; i < depth; i++) {
    accum += weight * perlin_noise(temp_p, pdata);
    weight *= 0.5f;
    temp_p = 2.0f * temp_p;
  }
  return fabsf(accum);
}

__device__ vec3_gpu get_texture_color(const TextureGPU& tex, float u, float v,
                                      const point3_gpu& p,
                                      const PerlinDataGPU* perlin,
                                      unsigned char* image_data) {
  if (tex.type == TextureType::SOLID) {
    return to_gpu(tex.solid.color);
  } else if (tex.type == TextureType::CHECKER) {
    int x = int(floorf(tex.checker.inv_scale * p.x()));
    int y = int(floorf(tex.checker.inv_scale * p.y()));
    int z = int(floorf(tex.checker.inv_scale * p.z()));
    return ((x + y + z) % 2 == 0) ? vec3_gpu(0.2, 0.3, 0.1) : vec3_gpu(0.9, 0.9, 0.9);
  } else if (tex.type == TextureType::IMAGE) {
    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = 1.0f - fminf(fmaxf(v, 0.0f), 1.0f);
    int i = int(u * tex.image.width);
    int j = int(v * tex.image.height);
    if (i >= tex.image.width) i = tex.image.width - 1;
    if (j >= tex.image.height) j = tex.image.height - 1;
    float r = image_data[tex.image.offset + 3 * (i + j * tex.image.width)] / 255.0f;
    float g = image_data[tex.image.offset + 3 * (i + j * tex.image.width) + 1] / 255.0f;
    float b = image_data[tex.image.offset + 3 * (i + j * tex.image.width) + 2] / 255.0f;
    return vec3_gpu(r, g, b);
  } else if (tex.type == TextureType::NOISE) {
    return vec3_gpu(1, 1, 1) * 0.5f * (1.0f + sinf(tex.noise.scale * p.z() + 10.0f * perlin_turb(p, perlin[tex.noise.perlin_data_idx], 7)));
  }
  return vec3_gpu(0, 0, 0);
}

__global__ void shade_kernel(PathStateSOA paths, HitResultSOA hits,
                             int* active_indices, int num_active,
                             MaterialGPU* materials, TextureGPU* textures,
                             PerlinDataGPU* perlin, unsigned char* images,
                             curandState* rand_state, int* next_active,
                             int* next_count, camera_gpu cam) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_active) return;
  int path_idx = active_indices[idx];
  if (!paths.alive[path_idx]) return;

  if (hits.hit_anything[idx]) {
    MaterialGPU mat = materials[hits.hit_material_id[idx]];
    vec3_gpu attenuation;
    ray_gpu scattered;
    point3_gpu rec_p = to_gpu(hits.hit_p[idx]);
    vec3_gpu rec_normal = to_gpu(hits.hit_normal[idx]);
    float rec_u = hits.hit_u[idx], rec_v = hits.hit_v[idx];
    bool rec_front_face = hits.hit_front_face[idx];

    if (mat.type == MaterialType::DIFFUSE_LIGHT) {
        paths.color[path_idx] += paths.attenuation[path_idx] * get_texture_color(textures[mat.albedo_tex_id], rec_u, rec_v, rec_p, perlin, images);
        paths.alive[path_idx] = false;
        return;
    }

    curandState local_rand = rand_state[path_idx];
    bool is_scattered = false;
    if (mat.type == MaterialType::LAMBERTIAN) {
      vec3_gpu scatter_dir = rec_normal + random_unit_vector(&local_rand);
      if (scatter_dir.length_squared() < 1e-8f) scatter_dir = rec_normal;
      scattered = ray_gpu(rec_p, scatter_dir, paths.ray_time[path_idx]);
      attenuation = get_texture_color(textures[mat.albedo_tex_id], rec_u, rec_v, rec_p, perlin, images);
      is_scattered = true;
    } else if (mat.type == MaterialType::METAL) {
        vec3_gpu reflected = reflect(unit_vector(paths.ray_dir[path_idx]), rec_normal);
        scattered = ray_gpu(rec_p, reflected + mat.fuzz * random_unit_vector(&local_rand), paths.ray_time[path_idx]);
        attenuation = get_texture_color(textures[mat.albedo_tex_id], rec_u, rec_v, rec_p, perlin, images);
        is_scattered = (dot(scattered.direction(), rec_normal) > 0);
    } else if (mat.type == MaterialType::DIELECTRIC) {
        attenuation = vec3_gpu(1.0, 1.0, 1.0);
        float refraction_ratio = rec_front_face ? (1.0f / mat.ref_idx) : mat.ref_idx;
        vec3_gpu unit_direction = unit_vector(paths.ray_dir[path_idx]);
        float cos_theta = fminf(dot(-unit_direction, rec_normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        vec3_gpu direction;
        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(&local_rand))
            direction = reflect(unit_direction, rec_normal);
        else
            direction = refract(unit_direction, rec_normal, refraction_ratio);
        scattered = ray_gpu(rec_p, direction, paths.ray_time[path_idx]);
        is_scattered = true;
    } else if (mat.type == MaterialType::ISOTROPIC) {
        scattered = ray_gpu(rec_p, random_unit_vector(&local_rand), paths.ray_time[path_idx]);
        attenuation = get_texture_color(textures[mat.albedo_tex_id], rec_u, rec_v, rec_p, perlin, images);
        is_scattered = true;
    }

    if (is_scattered) {
      paths.attenuation[path_idx] = paths.attenuation[path_idx] * attenuation;
      paths.ray_origin[path_idx] = scattered.origin();
      paths.ray_dir[path_idx] = scattered.direction();
      int next_idx = atomicAdd(next_count, 1);
      next_active[next_idx] = path_idx;
    } else {
      paths.alive[path_idx] = false;
    }
    rand_state[path_idx] = local_rand;
  } else {
    paths.color[path_idx] += paths.attenuation[path_idx] * cam.background;
    paths.alive[path_idx] = false;
  }
}

__global__ void accumulate(PathStateSOA paths, vec3_gpu* accum_buffer, int total_rays) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_rays) return;
  int pixel_index = paths.pixel_index[idx];
  atomicAdd(&accum_buffer[pixel_index].e[0], paths.color[idx].x());
  atomicAdd(&accum_buffer[pixel_index].e[1], paths.color[idx].y());
  atomicAdd(&accum_buffer[pixel_index].e[2], paths.color[idx].z());
}

__device__ vec3_gpu aces_tonemap(vec3_gpu x) {
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    vec3_gpu num = x * (a * x + b);
    vec3_gpu den = x * (c * x + d) + e;
    vec3_gpu res = num / den;
    return vec3_gpu(fmaxf(0.0f, fminf(1.0f, res.x())), fmaxf(0.0f, fminf(1.0f, res.y())), fmaxf(0.0f, fminf(1.0f, res.z())));
}

__global__ void finalize(float4* frame_buffer, vec3_gpu* accum_buffer, int total_pixels, int total_samples) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) return;
  vec3_gpu c = accum_buffer[idx] / float(total_samples);
  vec3_gpu tonemapped = aces_tonemap(c);
  frame_buffer[idx] = make_float4(powf(tonemapped.x(), 1.0f/2.2f), powf(tonemapped.y(), 1.0f/2.2f), powf(tonemapped.z(), 1.0f/2.2f), 1.0f);
}

static void alloc_path_state_soa(PathStateSOA& p, int n) {
  cudaMalloc(&p.ray_origin, n * sizeof(vec3_gpu)); cudaMalloc(&p.ray_dir, n * sizeof(vec3_gpu));
  cudaMalloc(&p.ray_time, n * sizeof(float)); cudaMalloc(&p.attenuation, n * sizeof(vec3_gpu));
  cudaMalloc(&p.color, n * sizeof(vec3_gpu)); cudaMalloc(&p.pixel_index, n * sizeof(int));
  cudaMalloc(&p.depth, n * sizeof(int)); cudaMalloc(&p.alive, n * sizeof(bool));
}

static void free_path_state_soa(PathStateSOA& p) {
  cudaFree(p.ray_origin); cudaFree(p.ray_dir); cudaFree(p.ray_time);
  cudaFree(p.attenuation); cudaFree(p.color); cudaFree(p.pixel_index);
  cudaFree(p.depth); cudaFree(p.alive);
}

static void alloc_hit_result_soa(HitResultSOA& h, int n) {
  cudaMalloc(&h.hit_p, n * sizeof(Vec3f)); cudaMalloc(&h.hit_normal, n * sizeof(Vec3f));
  cudaMalloc(&h.hit_t, n * sizeof(float)); cudaMalloc(&h.hit_u, n * sizeof(float));
  cudaMalloc(&h.hit_v, n * sizeof(float)); cudaMalloc(&h.hit_front_face, n * sizeof(bool));
  cudaMalloc(&h.hit_material_id, n * sizeof(int)); cudaMalloc(&h.hit_anything, n * sizeof(bool));
}

static void free_hit_result_soa(HitResultSOA& h) {
  cudaFree(h.hit_p); cudaFree(h.hit_normal); cudaFree(h.hit_t);
  cudaFree(h.hit_u); cudaFree(h.hit_v); cudaFree(h.hit_front_face);
  cudaFree(h.hit_material_id); cudaFree(h.hit_anything);
}

extern "C" void launch_render(RenderConfig config, BVHBuffer h_bvh, PrimitiveBuffer h_prims, MaterialBuffer h_mats,
                              TextureBuffer h_texs, PerlinBuffer h_perlin, ImageArrayBuffer h_images) {
  int width = config.width, height = config.height;
  int BATCH_SIZE = 16, total_rays = width * height * BATCH_SIZE;

  static curandState* d_rand_state = nullptr;
  static int last_w = 0, last_h = 0;
  if (!d_rand_state || width != last_w || height != last_h) {
    if (d_rand_state) cudaFree(d_rand_state);
    cudaMalloc(&d_rand_state, total_rays * sizeof(curandState));
    render_init<<<(total_rays+255)/256, 256>>>(total_rays, d_rand_state);
    last_w = width; last_h = height;
  }

  LinearBVHNode *d_bvh; PrimitiveGPU *d_prims; MaterialGPU *d_mats;
  TextureGPU *d_texs = nullptr; PerlinDataGPU *d_perlin = nullptr; unsigned char *d_imgs = nullptr;
  cudaMalloc(&d_bvh, h_bvh.count * sizeof(LinearBVHNode));
  cudaMalloc(&d_prims, h_prims.count * sizeof(PrimitiveGPU));
  cudaMalloc(&d_mats, h_mats.count * sizeof(MaterialGPU));
  cudaMemcpy(d_bvh, h_bvh.data, h_bvh.count * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prims, h_prims.data, h_prims.count * sizeof(PrimitiveGPU), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mats, h_mats.data, h_mats.count * sizeof(MaterialGPU), cudaMemcpyHostToDevice);
  if (h_texs.count > 0) { cudaMalloc(&d_texs, h_texs.count * sizeof(TextureGPU)); cudaMemcpy(d_texs, h_texs.data, h_texs.count * sizeof(TextureGPU), cudaMemcpyHostToDevice); }
  if (h_perlin.count > 0) { cudaMalloc(&d_perlin, h_perlin.count * sizeof(PerlinDataGPU)); cudaMemcpy(d_perlin, h_perlin.data, h_perlin.count * sizeof(PerlinDataGPU), cudaMemcpyHostToDevice); }
  if (h_images.count_bytes > 0) { cudaMalloc(&d_imgs, h_images.count_bytes); cudaMemcpy(d_imgs, h_images.data, h_images.count_bytes, cudaMemcpyHostToDevice); }

  camera_gpu cam;
  cam.aspect_ratio = float(width)/height; cam.image_width = width;
  cam.lookfrom = to_gpu(config.lookfrom); cam.lookat = to_gpu(config.lookat); cam.vup = to_gpu(config.vup);
  cam.vfov = config.vfov; cam.defocus_angle = config.defocus_angle; cam.focus_dist = config.focus_dist;
  cam.background = to_gpu(config.background); cam.initialize();

  static vec3_gpu* d_accum = nullptr;
  static int last_acc_sz = 0;
  if (!d_accum || width*height != last_acc_sz) {
    if (d_accum) cudaFree(d_accum);
    cudaMalloc(&d_accum, width*height*sizeof(vec3_gpu));
    last_acc_sz = width*height;
  }
  cudaMemset(d_accum, 0, width*height*sizeof(vec3_gpu));

  PathStateSOA d_paths; HitResultSOA d_hits;
  alloc_path_state_soa(d_paths, total_rays); alloc_hit_result_soa(d_hits, total_rays);
  int *d_active, *d_next, *d_cnt;
  cudaMalloc(&d_active, total_rays*sizeof(int)); cudaMalloc(&d_next, total_rays*sizeof(int)); cudaMalloc(&d_cnt, sizeof(int));

  int batches = (config.samples_per_pixel + BATCH_SIZE - 1) / BATCH_SIZE;
  for (int b = 0; b < batches; b++) {
    int cur = std::min(BATCH_SIZE, config.samples_per_pixel - b*BATCH_SIZE);
    int active = width * height * cur;
    thrust::sequence(thrust::device, d_active, d_active + active);
    generate_rays<<<(active+255)/256, 256>>>(d_paths, cam, d_rand_state, width, height, cur);
    for (int bounce = 0; bounce < config.max_depth && active > 0; bounce++) {
        cudaMemset(d_hits.hit_anything, 0, active*sizeof(bool));
        intersect_bvh<<<(active+255)/256, 256>>>(d_paths, d_hits, d_active, active, d_bvh, d_prims, d_rand_state);
        cudaMemset(d_cnt, 0, sizeof(int));
        shade_kernel<<<(active+255)/256, 256>>>(d_paths, d_hits, d_active, active, d_mats, d_texs, d_perlin, d_imgs, d_rand_state, d_next, d_cnt, cam);
        cudaMemcpy(&active, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        std::swap(d_active, d_next);
    }
    accumulate<<<(width*height*cur+255)/256, 256>>>(d_paths, d_accum, width*height*cur);
  }

  finalize<<<(width*height+255)/256, 256>>>((float4*)config.frame_buffer, d_accum, width*height, config.samples_per_pixel);
  cudaDeviceSynchronize();

  free_path_state_soa(d_paths); free_hit_result_soa(d_hits);
  cudaFree(d_active); cudaFree(d_next); cudaFree(d_cnt);
  cudaFree(d_bvh); cudaFree(d_prims); cudaFree(d_mats);
  if (d_texs) cudaFree(d_texs); if (d_perlin) cudaFree(d_perlin); if (d_imgs) cudaFree(d_imgs);
}

static cudaExternalMemory_t s_extMem = nullptr;

extern "C" void* import_vulkan_memory(int fd, size_t size) {
    if (s_extMem) { cudaDestroyExternalMemory(s_extMem); s_extMem = nullptr; }
    cudaExternalMemoryHandleDesc d = {}; d.type = cudaExternalMemoryHandleTypeOpaqueFd; d.handle.fd = fd; d.size = size;
    if (cudaImportExternalMemory(&s_extMem, &d) != cudaSuccess) return nullptr;
    cudaExternalMemoryBufferDesc bd = {}; bd.offset = 0; bd.size = size;
    void* p; if (cudaExternalMemoryGetMappedBuffer(&p, s_extMem, &bd) != cudaSuccess) return nullptr;
    return p;
}

extern "C" void cleanup_cuda_interop() {
    if (s_extMem) { cudaDestroyExternalMemory(s_extMem); s_extMem = nullptr; }
}
