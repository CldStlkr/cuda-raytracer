#include "cuda/camera.cuh"
#include "cuda/ray.cuh"
#include "cuda/sphere.cuh"
#include "cuda/vec.cuh"

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

__device__ vec3_gpu ray_color(const ray_gpu &r_in, sphere_gpu *spheres,
                              int num_spheres, curandState *local_rand_state,
                              int max_depth) {
  ray_gpu curr_ray = r_in;
  vec3_gpu curr_attenuation(1.0f, 1.0f, 1.0f);

  for (int depth = 0; depth < max_depth; ++depth) {
    // Find closest hit (only 1 sphere right now but will eventually have many)
    float closest_t = 1000.0f;
    int hit_idx = -1;
    float temp_t;

    for (int i = 0; i < num_spheres; ++i) {
      if (spheres[i].hit(curr_ray, 0.001f, closest_t, temp_t)) {
        closest_t = temp_t;
        hit_idx = i;
      }
    }

    if (hit_idx != -1) {
      // Hit a sphere
      // First, we need the exact point we hit in 3D space
      point3_gpu hit_point = curr_ray.at(closest_t);

      // Next, we need the surface normal at that point
      vec3_gpu normal = normalize(hit_point - spheres[hit_idx].get_center());

      sphere_gpu hit_sphere = spheres[hit_idx];

      if (hit_sphere.mat_type == MAT_DIFFUSE) {
        // --- DIFFUSE BEHAVIOR ---
        vec3_gpu target_direction =
            normal + random_unit_vector(local_rand_state);

        // Edge case: if random vector is exactly opposite normal,
        // target_directioncould be near zero
        if (target_direction.length_squared() < 0.0001f) {
          target_direction = normal;
        }

        // Update ray for next iteration of the loop
        curr_ray = ray_gpu(hit_point, target_direction);

        curr_attenuation = curr_attenuation * 0.5f;
      } else if (hit_sphere.mat_type == MAT_METAL) {
        // --- METAL BEHAVIOR ---
        vec3_gpu reflected = reflect(normalize(curr_ray.direction()), normal);

        // Add fuzz
        vec3_gpu target_direction =
            reflected +
            hit_sphere.fuzz * random_in_unit_sphere(local_rand_state);
        curr_ray = ray_gpu(hit_point, target_direction);
        curr_attenuation = curr_attenuation * hit_sphere.albedo;

        // If fuzzed ray scattered below surface (dot product < 0)
        // metal absorbs it completely (stops bouncing)
        if (dot(target_direction, normal) <= 0.0f) {
          return vec3_gpu(0, 0, 0);
        }
      } else if (hit_sphere.mat_type == MAT_DIELECTRIC) {
        // --- DIELECTRICT BEHAVIOR ---
        curr_attenuation = curr_attenuation * hit_sphere.albedo;

        // Entering or exiting the glass?
        float refraction_ratio;
        vec3_gpu outward_normal;

        // If the reay and normal are in the same diretion, the ray is inside
        // trying to get out. Flip normal so it points inward.
        if (dot(curr_ray.direction(), normal) > 0.0f) {
          outward_normal = -normal;
          refraction_ratio = hit_sphere.ir; // Glass to Air
        } else {
          outward_normal = normal;
          refraction_ratio = 1.0f / hit_sphere.ir; // Air to Glass
        }

        vec3_gpu unit_direction = normalize(curr_ray.direction());

        // Trig to check for Total Internal Reflection
        float cos_theta = fminf(dot(-unit_direction, outward_normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

        vec3_gpu target_direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) >
                                  curand_uniform(local_rand_state)) {
          target_direction = reflect(unit_direction, outward_normal);
        } else {
          target_direction =
              refract(unit_direction, outward_normal, refraction_ratio);
        }

        curr_ray = ray_gpu(hit_point, target_direction);
      }

    } else {
      // We missed (Hit sky)
      vec3_gpu unit_direction = normalize(curr_ray.direction());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      vec3_gpu sky_color = (1.0f - t) * vec3_gpu(1.0f, 1.0f, 1.0f) +
                           t * vec3_gpu(0.5f, 0.7f, 1.0f);

      // Final color is the sky color, modified by whatever surfaces we bounced
      // off previously
      return curr_attenuation * sky_color;
    }
  }

  // If we exceed max_depth bounces, assume all light was absorbed
  return vec3_gpu(0.0f, 0.0f, 0.0f);
}
__global__ void render_kernel(vec3_gpu *frame_buffer, int width, int height,
                              sphere_gpu *spheres, int num_spheres,
                              camera_gpu cam, curandState *rand_state,
                              int samples_per_pixel) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

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
    col += ray_color(r, spheres, num_spheres, &local_rand_state, 10);
  }

  // 4. Average
  col /= float(samples_per_pixel);

  // Save state back (important for future frames!)
  rand_state[pixel_index] = local_rand_state;

  frame_buffer[pixel_index] = col;
}

extern "C" void launch_render(vec3_gpu *frame_buffer, int width, int height,
                              int samples_per_pixel) {
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
  // ----------------

  int num_spheres = 20;
  sphere_gpu h_spheres[20];
  sphere_gpu *d_spheres;

  // Ground sphere
  h_spheres[0] =
      sphere_gpu(vec3_gpu(0, -1000, 0), 1000, vec3_gpu(0.5, 0.5, 0.5));

  // Central large glass sphere
  h_spheres[1] = sphere_gpu(vec3_gpu(0, 1, 0), 1.0, 1.5);

  // Surrounding spheres in a circle pattern
  h_spheres[2] = sphere_gpu(vec3_gpu(-5, 1, 0), 1.0, vec3_gpu(0.7, 0.2, 0.2));
  h_spheres[3] = sphere_gpu(vec3_gpu(5, 1, 0), 1.0, vec3_gpu(0.2, 0.2, 0.7));
  h_spheres[4] = sphere_gpu(vec3_gpu(0, 1, -5), 1.0, vec3_gpu(0.2, 0.7, 0.2));
  h_spheres[5] = sphere_gpu(vec3_gpu(0, 1, 5), 1.0, vec3_gpu(0.7, 0.7, 0.2));

  // Metal spheres at diagonal positions
  h_spheres[6] =
      sphere_gpu(vec3_gpu(-3.5, 1, -3.5), 1.0, vec3_gpu(0.8, 0.6, 0.2), 0.0);
  h_spheres[7] =
      sphere_gpu(vec3_gpu(3.5, 1, 3.5), 1.0, vec3_gpu(0.8, 0.8, 0.9), 0.1);
  h_spheres[8] =
      sphere_gpu(vec3_gpu(-3.5, 1, 3.5), 1.0, vec3_gpu(0.7, 0.4, 0.3), 0.2);
  h_spheres[9] =
      sphere_gpu(vec3_gpu(3.5, 1, -3.5), 1.0, vec3_gpu(0.9, 0.9, 0.9), 0.0);

  // Smaller spheres at different heights
  h_spheres[10] =
      sphere_gpu(vec3_gpu(-2, 0.5, -2), 0.5, vec3_gpu(0.6, 0.2, 0.6));
  h_spheres[11] = sphere_gpu(vec3_gpu(2, 0.5, 2), 0.5, vec3_gpu(0.8, 0.4, 0.1));
  h_spheres[12] =
      sphere_gpu(vec3_gpu(-2, 0.5, 2), 0.5, vec3_gpu(0.2, 0.6, 0.6));
  h_spheres[13] =
      sphere_gpu(vec3_gpu(2, 0.5, -2), 0.5, vec3_gpu(0.8, 0.4, 0.6));

  // Some elevated spheres for depth
  h_spheres[14] = sphere_gpu(vec3_gpu(-1, 2, -1), 0.3, vec3_gpu(0.9, 0.9, 0.9));
  h_spheres[15] = sphere_gpu(vec3_gpu(1, 2, 1), 0.3, vec3_gpu(0.1, 0.1, 0.1));

  // Glass spheres at different positions
  h_spheres[16] = sphere_gpu(vec3_gpu(-6, 0.7, -2), 0.7, 1.3);
  h_spheres[17] = sphere_gpu(vec3_gpu(6, 0.7, 2), 0.7, 1.8);

  // Far background spheres for depth
  h_spheres[18] =
      sphere_gpu(vec3_gpu(-10, 1.5, -8), 1.5, vec3_gpu(0.5, 0.5, 0.7), 0.3);
  h_spheres[19] =
      sphere_gpu(vec3_gpu(8, 1.2, -10), 1.2, vec3_gpu(0.4, 0.6, 0.4));

  CUDA_CHECK(cudaMalloc(&d_spheres, num_spheres * sizeof(sphere_gpu)));
  CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres, num_spheres * sizeof(sphere_gpu),
                        cudaMemcpyHostToDevice));
  camera_gpu cam;
  cam.aspect_ratio = float(width) / float(height);
  cam.image_width = width;
  cam.defocus_angle = 0.6f;
  cam.focus_dist = 10.0f;
  cam.initialize();

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  render_kernel<<<gridSize, blockSize>>>(frame_buffer, width, height, d_spheres,
                                         num_spheres, cam, d_rand_state,
                                         samples_per_pixel);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_spheres));
}
