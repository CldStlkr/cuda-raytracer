#ifndef CAMERA_CUD_H
#define CAMERA_CUD_H
#include "ray.cuh"
#include "vec.cuh"
#include <cstdio>
#include <cuda_runtime.h>

class camera_gpu {
public:
  float aspect_ratio = 1.0f;
  int image_width = 100;

  float vfov = 20.0f; // vertical field-of-view in degrees
  point3_gpu lookfrom = point3_gpu(13.0f, 2.0f, 3.0f);
  point3_gpu lookat = point3_gpu(0.0f, 0.0f, 0.0f);
  vec3_gpu vup = vec3_gpu(0.0f, 1.0f, 0.0f);

  float defocus_angle = 0.0f; // Variation angle of rays through each pixel
  float focus_dist =
      10.0f; // Distance from camera lookfrom point to plane of perfect focus

  float time0 = 0.0f;
  float time1 = 0.0f;
  vec3_gpu background;

  __host__ __device__ camera_gpu() {}

  __host__ void initialize() {
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    center = lookfrom;

    float theta = vfov * (M_PI / 180.0f);
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h * focus_dist;
    float viewport_width =
        viewport_height * (float(image_width) / image_height);

    vec3_gpu w = normalize(lookfrom - lookat);
    vec3_gpu u = normalize(cross(vup, w));
    vec3_gpu v = cross(w, u);

    vec3_gpu viewport_u = viewport_width * u;
    vec3_gpu viewport_v = viewport_height * -v;

    pixel_delta_u = viewport_u / float(image_width);
    pixel_delta_v = viewport_v / float(image_height);

    point3_gpu viewport_upper_left =
        center - (focus_dist * w) - viewport_u / 2.0f - viewport_v / 2.0f;
    pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

    // Calculate the camera defocus disk basis vectors
    float defocus_radius =
        focus_dist * tanf((defocus_angle / 2.0f) * (M_PI / 180.0f));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
  }

  // get_ray now takes the random state
  __device__ ray_gpu get_ray(float s, float t,
                             curandState *local_rand_state) const {
    point3_gpu pixel = pixel00_loc + s * pixel_delta_u + t * pixel_delta_v;

    // Shoot rays from a random point on the lens disk
    point3_gpu ray_origin;
    if (defocus_angle <= 0.0f) {
      ray_origin = center;
    } else {
      vec3_gpu p = random_in_unit_disk(local_rand_state);
      vec3_gpu offset = p.x() * defocus_disk_u + p.y() * defocus_disk_v;
      ray_origin = center + offset;
    }

    vec3_gpu dir = pixel - ray_origin;
    float ray_time = time0 + curand_uniform(local_rand_state) * (time1 - time0);
    return ray_gpu(ray_origin, dir, ray_time);
  }

  __host__ int getImageHeight() const { return image_height; }

private:
  int image_height;
  point3_gpu center;
  point3_gpu pixel00_loc;
  vec3_gpu pixel_delta_u;
  vec3_gpu pixel_delta_v;
  vec3_gpu defocus_disk_u; // NEW
  vec3_gpu defocus_disk_v; // NEW
};

#endif
