#ifndef CAMERA_CUD_H
#define CAMERA_CUD_H
#include "ray.cuh"
#include "vec.cuh"
#include <cstdio>
#include <cuda_runtime.h>

class camera_gpu {
public:
  float aspect_ratio = 1.0f; // Ratio of image width over height
  int image_width = 100;     // Rendered image width in pixels

  __host__ __device__ camera_gpu() {}

  __host__ void initialize() {
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    center = point3_gpu(0.0f, 0.0f, 0.0f);

    float focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width =
        viewport_height * (float(image_width) / image_height);

    auto viewport_u = vec3_gpu(viewport_width, 0.0f, 0.0f);
    auto viewport_v = vec3_gpu(0.0f, -viewport_height, 0.0f);

    pixel_delta_u = viewport_u / float(image_width);
    pixel_delta_v = viewport_v / float(image_height);

    // Calculate pixel00_loc so that the CENTER pixel is at (0, 0,
    // -focal_length)
    point3_gpu center_pixel_target = point3_gpu(0.0f, 0.0f, -focal_length);
    float half_width = float(image_width) / 2.0f;
    float half_height = float(image_height) / 2.0f;
    pixel00_loc = center_pixel_target - half_width * pixel_delta_u -
                  half_height * pixel_delta_v;

    printf("Truly centered camera: pixel00_loc=(%.3f,%.3f,%.3f)\n",
           pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z());
  }

  // __host__ __device__ ray_gpu get_ray(float u, float v) const {
  //   point3_gpu pixel = pixel00_loc + u * pixel_delta_u + v * pixel_delta_v;
  //   // vec3_gpu dir = unit_vector(pixel - center); // Normalize the direction
  //   vec3_gpu dir = pixel - center;
  //   return ray_gpu(center, dir);
  // }
  //
  __host__ __device__ ray_gpu get_ray(float u, float v) const {
    point3_gpu pixel = pixel00_loc + u * pixel_delta_u + v * pixel_delta_v;
    vec3_gpu dir = pixel - center;
    return ray_gpu(center, dir);
  }

  __host__ int getImageHeight() const { return image_height; }

private:
  int image_height;       // Rendered image height
  point3_gpu center;      // Camera center
  point3_gpu pixel00_loc; // Location of pixel 0,0
  vec3_gpu pixel_delta_u; // Pixel-to-pixel horizontal step
  vec3_gpu pixel_delta_v; // Pixel-to-pixel vertical step
};

#endif
