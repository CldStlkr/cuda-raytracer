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

// Warp-aggregated atomic enqueue: 1 atomicAdd per warp instead of 32.
// Threads whose bit is set in `mask` push `value` into `queue`.
__device__ void warp_enqueue(int* queue, int* count, int value,
                             unsigned int mask) {
  int lane = threadIdx.x & 31;
  if (!(mask & (1u << lane))) return;
  int leader = __ffs(mask) - 1;
  int n = __popc(mask);
  int base;
  if (lane == leader) base = atomicAdd(count, n);
  base = __shfl_sync(mask, base, leader);
  int offset = __popc(mask & ((1u << lane) - 1));
  queue[base + offset] = value;
}

__global__ __launch_bounds__(256, 2) void generate_rays(
    PathStateSOA paths, camera_gpu cam,
    curandState* rand_state, // Array of size 'total_rays'
    int width, int height, int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_rays = width * height * batch_size;
  if (idx >= total_rays) return;

  int pixel_index = idx / batch_size;
  int i = pixel_index % width;
  int j = pixel_index / width;

  // Each ray has its own RNG state
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

  // Each thread gets same seed, different sequence number, no offset
  curand_init(1984, idx, 0, &rand_state[idx]);
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

  int i = int(floorf(p.x()));
  int j = int(floorf(p.y()));
  int k = int(floorf(p.z()));

  vec3_gpu c[2][2][2];

  for (int di = 0; di < 2; di++)
    for (int dj = 0; dj < 2; dj++)
      for (int dk = 0; dk < 2; dk++) {
        c[di][dj][dk] =
            make_vec3_gpu(pdata.randvec[pdata.perm_x[(i + di) & 255] ^
                                        pdata.perm_y[(j + dj) & 255] ^
                                        pdata.perm_z[(k + dk) & 255]]);
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
    temp_p *= 2.0f;
  }
  return fabsf(accum);
}

__device__ vec3_gpu eval_texture(const TextureGPU* tex_array, int root_tex_id,
                                 float u, float v, const point3_gpu& p,
                                 const unsigned char* image_buffer,
                                 const PerlinDataGPU* perlin_buffer) {
  int current_tex_id = root_tex_id;
  for (int iter = 0; iter < 4; ++iter) {
    if (current_tex_id < 0) return vec3_gpu(0.0f, 0.0f, 0.0f);
    TextureGPU tex = tex_array[current_tex_id];
    if (tex.type == TextureType::SOLID) {
      return make_vec3_gpu(tex.solid.color);
    } else if (tex.type == TextureType::CHECKER) {
      float x = floorf(tex.checker.inv_scale * p.x());
      float y = floorf(tex.checker.inv_scale * p.y());
      float z = floorf(tex.checker.inv_scale * p.z());
      bool isEven = int(x + y + z) % 2 == 0;
      current_tex_id =
          isEven ? tex.checker.even_tex_idx : tex.checker.odd_tex_idx;
    } else if (tex.type == TextureType::IMAGE) {
      if (tex.image.width <= 0) return vec3_gpu(0.0f, 1.0f, 1.0f);
      float uu = fminf(fmaxf(u, 0.0f), 1.0f);
      float vv = 1.0f - fminf(fmaxf(v, 0.0f), 1.0f);
      int i = int(uu * tex.image.width);
      int j = int(vv * tex.image.height);
      if (i >= tex.image.width) i = tex.image.width - 1;
      if (j >= tex.image.height) j = tex.image.height - 1;
      int pixel_idx =
          tex.image.offset + (j * tex.image.bytes_per_scanline) + (i * 3);
      return vec3_gpu(image_buffer[pixel_idx + 0] / 255.0f,
                      image_buffer[pixel_idx + 1] / 255.0f,
                      image_buffer[pixel_idx + 2] / 255.0f);
    } else if (tex.type == TextureType::NOISE) {
      const PerlinDataGPU& pdata = perlin_buffer[tex.noise.perlin_data_idx];
      float turb = perlin_turb(p * tex.noise.scale, pdata, 7);
      float arg = tex.noise.scale * p.z() + 10.0f * turb;
      float noise_val = 0.5f * (1.0f + sinf(arg));
      // clamp due to turb
      noise_val = fminf(fmaxf(noise_val, 0.0f), 1.0f);
      return vec3_gpu(0.5f * noise_val, 0.5f * noise_val, 0.5f * noise_val);
    }
  }
  return vec3_gpu(0.0f, 0.0f, 0.0f);
}

__global__ __launch_bounds__(256, 2) void intersect_rays(
    PathStateSOA paths, HitResultSOA hits, const int* active_indices,
    int num_active, const LinearBVHNode* bvh_nodes,
    const PrimitiveGPU* primitives, curandState* rand_state) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_active) return;

  int path_idx = active_indices[tid];

  // Reconstruct ray from SoA fields
  ray_gpu r(paths.ray_origin[path_idx], paths.ray_dir[path_idx],
            paths.ray_time[path_idx]);

  HitRecordGPU rec;
  bool did_hit = hit_linear_bvh(bvh_nodes, primitives, r, 0.001f, 9999.0f, rec,
                                &rand_state[path_idx]);

  // Write hit result into SoA
  hits.hit_p[path_idx] = rec.p;
  hits.hit_normal[path_idx] = rec.normal;
  hits.hit_t[path_idx] = rec.t;
  hits.hit_u[path_idx] = rec.u;
  hits.hit_v[path_idx] = rec.v;
  hits.hit_front_face[path_idx] = rec.front_face;
  hits.hit_material_id[path_idx] = rec.material_id;
  hits.hit_anything[path_idx] = did_hit;
}

__global__ void classify_and_enqueue(PathStateSOA paths,
                                     const HitResultSOA hits,
                                     const int* active_indices, int num_active,
                                     const MaterialGPU* mats,
                                     MaterialQueues queues, camera_gpu cam,
                                     int max_depth) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_active) return;
  int path_idx = active_indices[tid];

  if (!hits.hit_anything[path_idx]) {
    paths.color[path_idx] += paths.attenuation[path_idx] * cam.background;
    paths.alive[path_idx] = false;
    return;
  }

  int cur_depth = paths.depth[path_idx] + 1;
  paths.depth[path_idx] = cur_depth;
  if (cur_depth >= max_depth) {
    paths.alive[path_idx] = false;
    return;
  }

  // __activemask() must be called AFTER early returns so only surviving
  // threads participate in the ballot.
  unsigned int active = __activemask();
  int mt = (int)mats[hits.hit_material_id[path_idx]].type;
  unsigned int m0 = __ballot_sync(active, mt == 0);
  unsigned int m1 = __ballot_sync(active, mt == 1);
  unsigned int m2 = __ballot_sync(active, mt == 2);
  unsigned int m3 = __ballot_sync(active, mt == 3);
  unsigned int m4 = __ballot_sync(active, mt == 4);
  warp_enqueue(queues.queues[0], &queues.counts[0], path_idx, m0);
  warp_enqueue(queues.queues[1], &queues.counts[1], path_idx, m1);
  warp_enqueue(queues.queues[2], &queues.counts[2], path_idx, m2);
  warp_enqueue(queues.queues[3], &queues.counts[3], path_idx, m3);
  warp_enqueue(queues.queues[4], &queues.counts[4], path_idx, m4);
}

__global__ __launch_bounds__(256, 2) void shade_lambertian(
    PathStateSOA p, const HitResultSOA h, const int* q, int count,
    const MaterialGPU* mats, const TextureGPU* texs,
    const PerlinDataGPU* perlin, const unsigned char* images, curandState* rs,
    MaterialQueues queues) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  int pi = q[tid];
  curandState lr = rs[pi];
  point3_gpu hp = make_vec3_gpu(h.hit_p[pi]);
  vec3_gpu n = make_vec3_gpu(h.hit_normal[pi]);
  MaterialGPU mat = mats[h.hit_material_id[pi]];
  vec3_gpu albedo = eval_texture(texs, mat.albedo_tex_id, h.hit_u[pi],
                                 h.hit_v[pi], hp, images, perlin);
  vec3_gpu dir = n + random_unit_vector(&lr);
  if (dir.length_squared() < 0.0001f) dir = n;
  p.ray_origin[pi] = hp;
  p.ray_dir[pi] = dir;
  p.attenuation[pi] = p.attenuation[pi] * albedo;
  rs[pi] = lr;
  unsigned int mask = __activemask();
  warp_enqueue(queues.next_active, queues.next_count, pi, mask);
}

__global__ __launch_bounds__(256, 2) void shade_metal(
    PathStateSOA p, const HitResultSOA h, const int* q, int count,
    const MaterialGPU* mats, const TextureGPU* texs,
    const PerlinDataGPU* perlin, const unsigned char* images, curandState* rs,
    MaterialQueues queues) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  int pi = q[tid];
  curandState lr = rs[pi];
  point3_gpu hp = make_vec3_gpu(h.hit_p[pi]);
  vec3_gpu n = make_vec3_gpu(h.hit_normal[pi]);
  MaterialGPU mat = mats[h.hit_material_id[pi]];
  vec3_gpu albedo = eval_texture(texs, mat.albedo_tex_id, h.hit_u[pi],
                                 h.hit_v[pi], hp, images, perlin);
  vec3_gpu refl = reflect(normalize(p.ray_dir[pi]), n);
  vec3_gpu dir = refl + mat.fuzz * random_in_unit_sphere(&lr);
  rs[pi] = lr;
  bool ok = dot(dir, n) > 0.0f;
  if (ok) {
    p.ray_origin[pi] = hp;
    p.ray_dir[pi] = dir;
    p.attenuation[pi] = p.attenuation[pi] * albedo;
  } else {
    p.alive[pi] = false;
  }
  unsigned int active = __activemask();
  unsigned int alive_mask = __ballot_sync(active, ok);
  warp_enqueue(queues.next_active, queues.next_count, pi, alive_mask);
}

__global__ __launch_bounds__(256, 2) void shade_dielectric(
    PathStateSOA p, const HitResultSOA h, const int* q, int count,
    const MaterialGPU* mats, const TextureGPU* texs,
    const PerlinDataGPU* perlin, const unsigned char* images, curandState* rs,
    MaterialQueues queues) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  int pi = q[tid];
  curandState lr = rs[pi];
  point3_gpu hp = make_vec3_gpu(h.hit_p[pi]);
  vec3_gpu n = make_vec3_gpu(h.hit_normal[pi]);
  MaterialGPU mat = mats[h.hit_material_id[pi]];
  vec3_gpu albedo = eval_texture(texs, mat.albedo_tex_id, h.hit_u[pi],
                                 h.hit_v[pi], hp, images, perlin);
  p.attenuation[pi] = p.attenuation[pi] * albedo;
  float rr = h.hit_front_face[pi] ? (1.0f / mat.ref_idx) : mat.ref_idx;
  vec3_gpu ud = normalize(p.ray_dir[pi]);
  float ct = fminf(dot(-ud, n), 1.0f);
  float st = sqrtf(1.0f - ct * ct);
  vec3_gpu dir;
  if (rr * st > 1.0f || reflectance(ct, rr) > curand_uniform(&lr))
    dir = reflect(ud, n);
  else
    dir = refract(ud, n, rr);
  p.ray_origin[pi] = hp;
  p.ray_dir[pi] = dir;
  rs[pi] = lr;
  unsigned int mask = __activemask();
  warp_enqueue(queues.next_active, queues.next_count, pi, mask);
}

__global__ void
shade_diffuse_light(PathStateSOA p, const HitResultSOA h, const int* q,
                    int count, const MaterialGPU* mats, const TextureGPU* texs,
                    const PerlinDataGPU* perlin, const unsigned char* images) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  int pi = q[tid];
  point3_gpu hp = make_vec3_gpu(h.hit_p[pi]);
  MaterialGPU mat = mats[h.hit_material_id[pi]];
  vec3_gpu albedo = eval_texture(texs, mat.albedo_tex_id, h.hit_u[pi],
                                 h.hit_v[pi], hp, images, perlin);
  p.color[pi] += p.attenuation[pi] * albedo;
  p.alive[pi] = false;
}

__global__ __launch_bounds__(256, 2) void shade_isotropic(
    PathStateSOA p, const HitResultSOA h, const int* q, int count,
    const MaterialGPU* mats, const TextureGPU* texs,
    const PerlinDataGPU* perlin, const unsigned char* images, curandState* rs,
    MaterialQueues queues) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  int pi = q[tid];
  curandState lr = rs[pi];
  point3_gpu hp = make_vec3_gpu(h.hit_p[pi]);
  MaterialGPU mat = mats[h.hit_material_id[pi]];
  vec3_gpu albedo = eval_texture(texs, mat.albedo_tex_id, h.hit_u[pi],
                                 h.hit_v[pi], hp, images, perlin);
  p.ray_origin[pi] = hp;
  p.ray_dir[pi] = random_unit_vector(&lr);
  p.attenuation[pi] = p.attenuation[pi] * albedo;
  rs[pi] = lr;
  unsigned int mask = __activemask();
  warp_enqueue(queues.next_active, queues.next_count, pi, mask);
}

__global__ void accumulate(const PathStateSOA paths, vec3_gpu* frame_buffer,
                           int total_rays) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_rays) return;

  int pix = paths.pixel_index[idx];
  vec3_gpu c = paths.color[idx];
  atomicAdd(&frame_buffer[pix].e[0], c.x());
  atomicAdd(&frame_buffer[pix].e[1], c.y());
  atomicAdd(&frame_buffer[pix].e[2], c.z());
}

__global__ void finalize(vec3_gpu* frame_buffer, int total_pixels,
                         int total_samples) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) return;

  frame_buffer[idx] /= float(total_samples);
}

// Helper to allocate all SoA arrays for PathState
static void alloc_path_state_soa(PathStateSOA& p, int n) {
  cudaMalloc(&p.ray_origin, n * sizeof(vec3_gpu));
  cudaMalloc(&p.ray_dir, n * sizeof(vec3_gpu));
  cudaMalloc(&p.ray_time, n * sizeof(float));
  cudaMalloc(&p.attenuation, n * sizeof(vec3_gpu));
  cudaMalloc(&p.color, n * sizeof(vec3_gpu));
  cudaMalloc(&p.pixel_index, n * sizeof(int));
  cudaMalloc(&p.depth, n * sizeof(int));
  cudaMalloc(&p.alive, n * sizeof(bool));
}

static void free_path_state_soa(PathStateSOA& p) {
  cudaFree(p.ray_origin);
  cudaFree(p.ray_dir);
  cudaFree(p.ray_time);
  cudaFree(p.attenuation);
  cudaFree(p.color);
  cudaFree(p.pixel_index);
  cudaFree(p.depth);
  cudaFree(p.alive);
}

static void alloc_hit_result_soa(HitResultSOA& h, int n) {
  cudaMalloc(&h.hit_p, n * sizeof(Vec3f));
  cudaMalloc(&h.hit_normal, n * sizeof(Vec3f));
  cudaMalloc(&h.hit_t, n * sizeof(float));
  cudaMalloc(&h.hit_u, n * sizeof(float));
  cudaMalloc(&h.hit_v, n * sizeof(float));
  cudaMalloc(&h.hit_front_face, n * sizeof(bool));
  cudaMalloc(&h.hit_material_id, n * sizeof(int));
  cudaMalloc(&h.hit_anything, n * sizeof(bool));
}

static void free_hit_result_soa(HitResultSOA& h) {
  cudaFree(h.hit_p);
  cudaFree(h.hit_normal);
  cudaFree(h.hit_t);
  cudaFree(h.hit_u);
  cudaFree(h.hit_v);
  cudaFree(h.hit_front_face);
  cudaFree(h.hit_material_id);
  cudaFree(h.hit_anything);
}

extern "C" void launch_render(RenderConfig config, BVHBuffer h_bvh,
                              PrimitiveBuffer h_prims, MaterialBuffer h_mats,
                              TextureBuffer h_texs, PerlinBuffer h_perlin,
                              ImageArrayBuffer h_images) {
  int width = config.width, height = config.height;
  printf("CUDA: Starting render %dx%d\n", width, height);

  // Each batch processes BATCH_SIZE samples per pixel simultaneously.
  // Lower = less VRAM, but more batch iterations.
  int BATCH_SIZE = 16;
  int total_rays = width * height * BATCH_SIZE;

  // --- RNG SETUP ---
  static curandState* d_rand_state = nullptr;
  static int last_width = 0;
  static int last_height = 0;
  if (d_rand_state == nullptr || width != last_width || height != last_height) {
    if (d_rand_state) cudaFree(d_rand_state);
    CUDA_CHECK(
        cudaMalloc((void**)&d_rand_state, total_rays * sizeof(curandState)));

    dim3 blockSize(256);
    dim3 gridSize((total_rays + blockSize.x - 1) / blockSize.x);
    render_init<<<gridSize, blockSize>>>(total_rays, d_rand_state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    last_width = width;
    last_height = height;
    printf("CUDA: Initialized RNG for %d rays\n", total_rays);
  }

  LinearBVHNode* d_bvh_nodes;
  PrimitiveGPU* d_primitives;
  MaterialGPU* d_materials;
  TextureGPU* d_textures;
  PerlinDataGPU* d_perlin;
  unsigned char* d_images;

  size_t bvh_size = h_bvh.count * sizeof(LinearBVHNode);
  size_t prims_size = h_prims.count * sizeof(PrimitiveGPU);
  size_t mats_size = h_mats.count * sizeof(MaterialGPU);
  size_t texs_size = h_texs.count * sizeof(TextureGPU);
  size_t perlin_size = h_perlin.count * sizeof(PerlinDataGPU);
  size_t images_size = h_images.count_bytes * sizeof(unsigned char);

  CUDA_CHECK(cudaMalloc(&d_bvh_nodes, bvh_size));
  CUDA_CHECK(cudaMalloc(&d_primitives, prims_size));
  CUDA_CHECK(cudaMalloc(&d_materials, mats_size));
  if (texs_size > 0) CUDA_CHECK(cudaMalloc(&d_textures, texs_size));
  if (perlin_size > 0) CUDA_CHECK(cudaMalloc(&d_perlin, perlin_size));
  if (images_size > 0) CUDA_CHECK(cudaMalloc(&d_images, images_size));

  // Ship memory over PCIe bus
  CUDA_CHECK(
      cudaMemcpy(d_bvh_nodes, h_bvh.data, bvh_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_primitives, h_prims.data, prims_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_materials, h_mats.data, mats_size, cudaMemcpyHostToDevice));
  if (texs_size > 0)
    CUDA_CHECK(
        cudaMemcpy(d_textures, h_texs.data, texs_size, cudaMemcpyHostToDevice));
  if (perlin_size > 0)
    CUDA_CHECK(cudaMemcpy(d_perlin, h_perlin.data, perlin_size,
                          cudaMemcpyHostToDevice));
  if (images_size > 0)
    CUDA_CHECK(cudaMemcpy(d_images, h_images.data, images_size,
                          cudaMemcpyHostToDevice));

  camera_gpu cam;
  cam.aspect_ratio = float(width) / float(height);
  cam.image_width = width;
  cam.lookfrom =
      vec3_gpu(config.lookfrom.x, config.lookfrom.y, config.lookfrom.z);
  cam.lookat = vec3_gpu(config.lookat.x, config.lookat.y, config.lookat.z);
  cam.vup = vec3_gpu(config.vup.x, config.vup.y, config.vup.z);
  cam.vfov = config.vfov;
  cam.defocus_angle = config.defocus_angle;
  cam.focus_dist = config.focus_dist;
  cam.time0 = 0.0f;
  cam.time1 = 1.0f;
  cam.background =
      vec3_gpu(config.background.x, config.background.y, config.background.z);
  cam.initialize();

  // Wavefront SoA allocations
  PathStateSOA d_paths;
  HitResultSOA d_hits;
  alloc_path_state_soa(d_paths, total_rays);
  alloc_hit_result_soa(d_hits, total_rays);

  int* d_active_indices;
  CUDA_CHECK(cudaMalloc(&d_active_indices, total_rays * sizeof(int)));

  // Material queue allocations
  MaterialQueues queues;
  for (int i = 0; i < NUM_MATERIAL_TYPES; i++)
    CUDA_CHECK(cudaMalloc(&queues.queues[i], total_rays * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&queues.counts, NUM_MATERIAL_TYPES * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&queues.next_active, total_rays * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&queues.next_count, sizeof(int)));

  // Clear frame buffer before accumulating
  CUDA_CHECK(
      cudaMemset(config.frame_buffer, 0, width * height * sizeof(vec3_gpu)));

  int num_sample_batches =
      (config.samples_per_pixel + BATCH_SIZE - 1) / BATCH_SIZE;
  dim3 block1D(256);

  for (int batch = 0; batch < num_sample_batches; batch++) {
    int current_batch_size = (batch == num_sample_batches - 1 &&
                              config.samples_per_pixel % BATCH_SIZE != 0)
                                 ? config.samples_per_pixel % BATCH_SIZE
                                 : BATCH_SIZE;
    int active_rays = width * height * current_batch_size;

    thrust::sequence(thrust::device, d_active_indices,
                     d_active_indices + active_rays);

    dim3 gen_grid((active_rays + 255) / 256);
    generate_rays<<<gen_grid, block1D>>>(d_paths, cam, d_rand_state, width,
                                         height, current_batch_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int bounce = 0; bounce < config.max_depth && active_rays > 0;
         bounce++) {
      dim3 active_grid((active_rays + 255) / 256);

      intersect_rays<<<active_grid, block1D>>>(
          d_paths, d_hits, d_active_indices, active_rays, d_bvh_nodes,
          d_primitives, d_rand_state);

      // Reset queue counters
      cudaMemset(queues.counts, 0, NUM_MATERIAL_TYPES * sizeof(int));
      cudaMemset(queues.next_count, 0, sizeof(int));

      // Classify hits and enqueue into per-material queues
      classify_and_enqueue<<<active_grid, block1D>>>(
          d_paths, d_hits, d_active_indices, active_rays, d_materials, queues,
          cam, config.max_depth);

      // Copy queue counts to host (20-byte D2H transfer)
      int h_counts[NUM_MATERIAL_TYPES];
      cudaMemcpy(h_counts, queues.counts, sizeof(h_counts),
                 cudaMemcpyDeviceToHost);

      // Launch per-material shade kernels (branch-free, only if non-empty)
      if (h_counts[0] > 0) {
        dim3 g((h_counts[0] + 255) / 256);
        shade_lambertian<<<g, block1D>>>(
            d_paths, d_hits, queues.queues[0], h_counts[0], d_materials,
            d_textures, d_perlin, d_images, d_rand_state, queues);
      }
      if (h_counts[1] > 0) {
        dim3 g((h_counts[1] + 255) / 256);
        shade_metal<<<g, block1D>>>(d_paths, d_hits, queues.queues[1],
                                    h_counts[1], d_materials, d_textures,
                                    d_perlin, d_images, d_rand_state, queues);
      }
      if (h_counts[2] > 0) {
        dim3 g((h_counts[2] + 255) / 256);
        shade_dielectric<<<g, block1D>>>(
            d_paths, d_hits, queues.queues[2], h_counts[2], d_materials,
            d_textures, d_perlin, d_images, d_rand_state, queues);
      }
      if (h_counts[3] > 0) {
        dim3 g((h_counts[3] + 255) / 256);
        shade_diffuse_light<<<g, block1D>>>(d_paths, d_hits, queues.queues[3],
                                            h_counts[3], d_materials,
                                            d_textures, d_perlin, d_images);
      }
      if (h_counts[4] > 0) {
        dim3 g((h_counts[4] + 255) / 256);
        shade_isotropic<<<g, block1D>>>(
            d_paths, d_hits, queues.queues[4], h_counts[4], d_materials,
            d_textures, d_perlin, d_images, d_rand_state, queues);
      }

      // Next bounce: active_indices = next_active queue
      cudaMemcpy(&active_rays, queues.next_count, sizeof(int),
                 cudaMemcpyDeviceToHost);
      std::swap(d_active_indices, queues.next_active);
    }

    dim3 accum_grid((width * height * current_batch_size + 255) / 256);
    accumulate<<<accum_grid, block1D>>>(d_paths, config.frame_buffer,
                                        width * height * current_batch_size);
  }

  dim3 finalize_grid((width * height + 255) / 256);
  finalize<<<finalize_grid, block1D>>>(config.frame_buffer, width * height,
                                       config.samples_per_pixel);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Clean up
  free_path_state_soa(d_paths);
  free_hit_result_soa(d_hits);
  CUDA_CHECK(cudaFree(d_active_indices));
  for (int i = 0; i < NUM_MATERIAL_TYPES; i++)
    CUDA_CHECK(cudaFree(queues.queues[i]));
  CUDA_CHECK(cudaFree(queues.counts));
  CUDA_CHECK(cudaFree(queues.next_active));
  CUDA_CHECK(cudaFree(queues.next_count));

  CUDA_CHECK(cudaFree(d_bvh_nodes));
  CUDA_CHECK(cudaFree(d_primitives));
  CUDA_CHECK(cudaFree(d_materials));
  if (texs_size > 0) CUDA_CHECK(cudaFree(d_textures));
  if (perlin_size > 0) CUDA_CHECK(cudaFree(d_perlin));
  if (images_size > 0) CUDA_CHECK(cudaFree(d_images));
}
