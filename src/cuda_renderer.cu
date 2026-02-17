#include "cuda/camera.cuh"
#include "cuda/ray.cuh"
#include "cuda/sphere.cuh"
#include "cuda/vec.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <stdio.h>

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) return;
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

__device__ vec3_gpu ray_color(const ray_gpu& r, sphere_gpu* spheres,
                              int num_spheres) {
  float t;

  for (int i = 0; i < num_spheres; i++) {
    if (spheres[i].hit(r, 0.001f, 1000.0f, t)) {
      return vec3_gpu(1.0f, 0.0f, 0.0f); // Red for sphere hits
    }
  }

  // Sky gradient
  vec3_gpu unit_direction = normalize(r.direction());
  t = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - t) * vec3_gpu(1.0f, 1.0f, 1.0f) +
         t * vec3_gpu(0.5f, 0.7f, 1.0f);
}
__global__ void render_kernel(vec3_gpu* frame_buffer, int width, int height,
                              sphere_gpu* spheres, int num_spheres,
                              camera_gpu cam) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= width || j >= height) return;

  int pixel_index = j * width + i;
  if (pixel_index >= width * height) return;

  // Center the UV coordinates properly
  // Normalized coordinates (0..1) are wrong for get_ray which expects pixel
  // coordinates
  float u = float(i) + 0.5f;
  float v = float(j) + 0.5f;

  ray_gpu r = cam.get_ray(u, v);
  vec3_gpu col = ray_color(r, spheres, num_spheres);

  frame_buffer[pixel_index] = col;

  // In your render kernel, add this for the center pixel only:
  if (i == width / 2 && j == height / 2) {
    ray_gpu center_ray = cam.get_ray(float(width) / 2.0f, float(height) / 2.0f);
    vec3_gpu dir = center_ray.direction();
    printf("Center ray direction: (%.6f, %.6f, %.6f)\n", dir.x(), dir.y(),
           dir.z());
    // Force this pixel red for testing
    frame_buffer[pixel_index] = vec3_gpu(1.0f, 0.0f, 0.0f);
    return;
  }
}

extern "C" void launch_render(vec3_gpu* frame_buffer, int width, int height) {
  printf("CUDA: Starting render %dx%d\n", width, height);

  // Put sphere in front of the camera (z = -5.0)
  sphere_gpu h_sphere(vec3_gpu(0.0f, 0.0f, -5.0f), 1.0f);

  printf("CUDA: Created sphere at (%.2f, %.2f, %.2f) with radius %.2f\n",
         h_sphere.get_center().x(), h_sphere.get_center().y(),
         h_sphere.get_center().z(), h_sphere.get_radius());

  // Test the sphere hit function on CPU before sending to GPU
  ray_gpu test_ray(vec3_gpu(0.0f, 0.0f, 0.0f), vec3_gpu(0.0f, 0.0f, -1.0f));
  float test_t;
  if (h_sphere.hit(test_ray, 0.001f, 1000.0f, test_t)) {
    printf("CPU test: Small sphere hit at t=%.3f\n", test_t);
  } else {
    printf("CPU test: Small sphere MISSED!\n");
  }

  sphere_gpu* d_spheres;

  CUDA_CHECK(cudaMalloc(&d_spheres, sizeof(sphere_gpu)));
  CUDA_CHECK(cudaMemcpy(d_spheres, &h_sphere, sizeof(sphere_gpu),
                        cudaMemcpyHostToDevice));

  camera_gpu cam;
  cam.aspect_ratio = float(width) / float(height);
  cam.image_width = width;
  cam.initialize();

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  printf("CUDA: Grid size: %dx%d, Block size: %dx%d\n", gridSize.x, gridSize.y,
         blockSize.x, blockSize.y);

  render_kernel<<<gridSize, blockSize>>>(frame_buffer, width, height, d_spheres,
                                         1, cam);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("CUDA: Render completed successfully\n");
  CUDA_CHECK(cudaFree(d_spheres));
}
