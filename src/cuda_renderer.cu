#include "cuda/bvh_kernel.cuh"
#include "cuda/camera.cuh"
#include "cuda/ray.cuh"
#include "cuda/vec.cuh"

#include "cuda_structs.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <stdio.h>

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;

  // Each thread gets same seed, different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
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

__device__ float perlin_noise(const point3_gpu &p, const PerlinDataGPU &pdata) {
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

__device__ float perlin_turb(const point3_gpu &p, const PerlinDataGPU &pdata,
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

__device__ vec3_gpu eval_texture(const TextureGPU *tex_array, int root_tex_id,
                                 float u, float v, const point3_gpu &p,
                                 const unsigned char *image_buffer,
                                 const PerlinDataGPU *perlin_buffer) {
  int current_tex_id = root_tex_id;
  for (int iter = 0; iter < 4; ++iter) {
    if (current_tex_id < 0)
      return vec3_gpu(0.0f, 0.0f, 0.0f);
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
      if (tex.image.width <= 0)
        return vec3_gpu(0.0f, 1.0f, 1.0f);
      float uu = fminf(fmaxf(u, 0.0f), 1.0f);
      float vv = 1.0f - fminf(fmaxf(v, 0.0f), 1.0f);
      int i = int(uu * tex.image.width);
      int j = int(vv * tex.image.height);
      if (i >= tex.image.width)
        i = tex.image.width - 1;
      if (j >= tex.image.height)
        j = tex.image.height - 1;
      int pixel_idx =
          tex.image.offset + (j * tex.image.bytes_per_scanline) + (i * 3);
      return vec3_gpu(image_buffer[pixel_idx + 0] / 255.0f,
                      image_buffer[pixel_idx + 1] / 255.0f,
                      image_buffer[pixel_idx + 2] / 255.0f);
    } else if (tex.type == TextureType::NOISE) {
      const PerlinDataGPU &pdata = perlin_buffer[tex.noise.perlin_data_idx];
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

__device__ vec3_gpu
ray_color(const ray_gpu &r_in, const LinearBVHNode *bvh_nodes,
          const PrimitiveGPU *primitives, const MaterialGPU *materials,
          const TextureGPU *textures, const PerlinDataGPU *perlin,
          const unsigned char *images, curandState *local_rand_state,
          int max_depth, const camera_gpu &cam) {
  ray_gpu curr_ray = r_in;
  vec3_gpu curr_attenuation(1.0f, 1.0f, 1.0f);

  for (int depth = 0; depth < max_depth; ++depth) {
    HitRecordGPU rec;

    if (hit_linear_bvh(bvh_nodes, primitives, curr_ray, 0.001f, 9999.0f, rec,
                       local_rand_state)) {

      // Extract hit data from the struct
      point3_gpu hit_point = make_vec3_gpu(rec.p);
      vec3_gpu normal = make_vec3_gpu(rec.normal);

      MaterialGPU mat = materials[rec.material_id];
      vec3_gpu albedo = eval_texture(textures, mat.albedo_tex_id, rec.u, rec.v,
                                     hit_point, images, perlin);
      vec3_gpu emission(0.0f, 0.0f, 0.0f);

      if (mat.type == MaterialType::DIFFUSE_LIGHT) {
        emission = albedo;
        // Diffuse light does not scatter, return accumulated so far + emission!
        return curr_attenuation * emission;
      }

      if (mat.type == MaterialType::LAMBERTIAN) {
        vec3_gpu target_direction =
            normal + random_unit_vector(local_rand_state);
        if (target_direction.length_squared() < 0.0001f) {
          target_direction = normal;
        }
        curr_ray = ray_gpu(hit_point, target_direction);
        curr_attenuation = curr_attenuation * albedo;
      } else if (mat.type == MaterialType::METAL) {
        vec3_gpu reflected = reflect(normalize(curr_ray.direction()), normal);
        vec3_gpu target_direction =
            reflected + mat.fuzz * random_in_unit_sphere(local_rand_state);

        // If scattered below surface (absorbed)
        if (dot(target_direction, normal) > 0.0f) {
          curr_ray = ray_gpu(hit_point, target_direction);
          curr_attenuation = curr_attenuation * albedo;
        } else {
          return vec3_gpu(0.0f, 0.0f, 0.0f);
        }
      } else if (mat.type == MaterialType::DIELECTRIC) {
        curr_attenuation = curr_attenuation * albedo;

        float refraction_ratio =
            rec.front_face ? (1.0f / mat.ref_idx) : mat.ref_idx;
        vec3_gpu unit_direction = normalize(curr_ray.direction());

        float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        vec3_gpu target_direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) >
                                  curand_uniform(local_rand_state)) {
          target_direction = reflect(unit_direction, normal);
        } else {
          target_direction = refract(unit_direction, normal, refraction_ratio);
        }
        curr_ray = ray_gpu(hit_point, target_direction);
      }
    } else {
      // Missed: Ray hit the sky / background.
      return curr_attenuation * cam.background;
    }
  }
  // Exceeded max depth: all energy absorbed
  return vec3_gpu(0.0f, 0.0f, 0.0f);
}

__global__ void render_kernel(RenderConfig config, BVHBuffer bvh,
                              PrimitiveBuffer prims, MaterialBuffer mats,
                              TextureBuffer texs, PerlinBuffer perlin,
                              ImageArrayBuffer images, camera_gpu cam,
                              curandState *rand_state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  int width = config.width, height = config.height;
  int samples_per_pixel = config.samples_per_pixel;

  if (i >= width || j >= height)
    return;

  int pixel_index = j * width + i;

  // 1. Load RNG state for this pixel
  curandState local_rand_state = rand_state[pixel_index];
  vec3_gpu col(0.0f, 0.0f, 0.0f);

  // 2. Loop for MSAA
  for (int s = 0; s < samples_per_pixel; s++) {
    // 3. Jitter: u = i + random(0..1)
    float u = float(i) + curand_uniform(&local_rand_state);
    float v = float(j) + curand_uniform(&local_rand_state);

    // Note: We use raw 'u' here because cam.get_ray expects pixel coordinates
    // (it multiplies by pixel_delta internally)
    ray_gpu r = cam.get_ray(u, v, &local_rand_state);
    col += ray_color(r, bvh.data, prims.data, mats.data, texs.data, perlin.data,
                     images.data, &local_rand_state, config.max_depth, cam);
  }

  // 4. Average
  col /= float(samples_per_pixel);

  // Save state back (important for future frames!)
  rand_state[pixel_index] = local_rand_state;

  config.frame_buffer[pixel_index] = col;
}

extern "C" void launch_render(RenderConfig config, BVHBuffer h_bvh,
                              PrimitiveBuffer h_prims, MaterialBuffer h_mats,
                              TextureBuffer h_texs, PerlinBuffer h_perlin,
                              ImageArrayBuffer h_images) {
  int width = config.width, height = config.height;
  printf("CUDA: Starting render %dx%d\n", width, height);
  // --- RNG SETUP ---
  static curandState *d_rand_state = nullptr;
  static int last_width = 0;
  static int last_height = 0;
  int num_pixels = width * height;
  if (d_rand_state == nullptr || width != last_width || height != last_height) {
    if (d_rand_state)
      cudaFree(d_rand_state);
    CUDA_CHECK(
        cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    render_init<<<gridSize, blockSize>>>(width, height, d_rand_state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    last_width = width;
    last_height = height;
    printf("CUDA: Initialized RNG\n");
  }

  LinearBVHNode *d_bvh_nodes;
  PrimitiveGPU *d_primitives;
  MaterialGPU *d_materials;
  TextureGPU *d_textures;
  PerlinDataGPU *d_perlin;
  unsigned char *d_images;

  size_t bvh_size = h_bvh.count * sizeof(LinearBVHNode);
  size_t prims_size = h_prims.count * sizeof(PrimitiveGPU);
  size_t mats_size = h_mats.count * sizeof(MaterialGPU);
  size_t texs_size = h_texs.count * sizeof(TextureGPU);
  size_t perlin_size = h_perlin.count * sizeof(PerlinDataGPU);
  size_t images_size = h_images.count_bytes * sizeof(unsigned char);

  CUDA_CHECK(cudaMalloc(&d_bvh_nodes, bvh_size));
  CUDA_CHECK(cudaMalloc(&d_primitives, prims_size));
  CUDA_CHECK(cudaMalloc(&d_materials, mats_size));
  if (texs_size > 0)
    CUDA_CHECK(cudaMalloc(&d_textures, texs_size));
  if (perlin_size > 0)
    CUDA_CHECK(cudaMalloc(&d_perlin, perlin_size));
  if (images_size > 0)
    CUDA_CHECK(cudaMalloc(&d_images, images_size));

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

  BVHBuffer d_bvh = {d_bvh_nodes, h_bvh.count};
  PrimitiveBuffer d_prims = {d_primitives, h_prims.count};
  MaterialBuffer d_mats = {d_materials, h_mats.count};
  TextureBuffer d_texs = {d_textures, h_texs.count};
  PerlinBuffer d_per = {d_perlin, h_perlin.count};
  ImageArrayBuffer d_img = {d_images, h_images.count_bytes};

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

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  render_kernel<<<gridSize, blockSize>>>(config, d_bvh, d_prims, d_mats, d_texs,
                                         d_per, d_img, cam, d_rand_state);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_bvh_nodes));
  CUDA_CHECK(cudaFree(d_primitives));
  CUDA_CHECK(cudaFree(d_materials));
  if (texs_size > 0)
    CUDA_CHECK(cudaFree(d_textures));
  if (perlin_size > 0)
    CUDA_CHECK(cudaFree(d_perlin));
  if (images_size > 0)
    CUDA_CHECK(cudaFree(d_images));
}
