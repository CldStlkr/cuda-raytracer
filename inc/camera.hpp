#ifndef CAMERA_HPP
#define CAMERA_HPP
#include "color.hpp"
#include "hittable.hpp"
#include "material.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

class camera {
public:
  double aspect_ratio = 1.0;
  int image_width = 100;
  int samples_per_pixel = 10;
  int max_depth = 10;
  color background;
  double vfov = 90;
  point3 lookfrom = point3(0, 0, 0);
  point3 lookat = point3(0, 0, -1);
  vec3 vup = vec3(0, 1, 0);
  double defocus_angle = 0;
  double focus_dist = 10;

  // Rendering options
  bool enable_antialiasing = true;
  bool enable_shadows = true;
  bool enable_reflections = true;
  bool enable_refractions = true;

  // Render to buffer with progress tracking and real-time updates
  void render_to_buffer_with_progress(const hittable &world,
                                      std::vector<unsigned char> &buffer,
                                      std::mutex &buffer_mutex,
                                      std::atomic<float> &progress,
                                      const std::atomic<bool> &should_stop,
                                      std::atomic<bool> &texture_needs_update) {
    initialize();

    // Ensure buffer is properly sized
    {
      std::lock_guard<std::mutex> lock(buffer_mutex);
      buffer.resize(image_width * image_height * 3);
      std::fill(buffer.begin(), buffer.end(), 0); // Clear to black
    }

    int total_pixels = image_width * image_height;
    std::atomic<int> completed_pixels{0};
    std::atomic<int> pixels_since_update{0};
    const int update_frequency = std::max(1, total_pixels / 100); // 1%

    std::cout << "Rendering " << image_width << "x" << image_height << " with "
              << samples_per_pixel << " samples per pixel..." << std::endl;

    std::atomic<int> current_line{0};
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
      num_threads = 4;
    std::vector<std::thread> threads;

    auto worker = [&]() {
      std::vector<unsigned char> scanline_buffer(image_width * 3);
      while (!should_stop.load()) {
        int j = current_line.fetch_add(1);
        if (j >= image_height)
          break;

        int row_completed_pixels = 0;

        for (int i = 0; i < image_width && !should_stop.load(); i++) {
          color pixel_color(0, 0, 0);
          int sample_count = enable_antialiasing ? samples_per_pixel : 1;

          for (int sample = 0; sample < sample_count && !should_stop.load();
               sample++) {
            ray r = get_ray(i, j);
            pixel_color += ray_color(r, max_depth, world);
          }

          if (should_stop.load())
            break;

          double scale = 1.0 / sample_count;
          pixel_color *= scale;

          auto r_val = linear_to_gamma(pixel_color.x());
          auto g_val = linear_to_gamma(pixel_color.y());
          auto b_val = linear_to_gamma(pixel_color.z());

          static const interval intensity(0.000, 0.999);
          scanline_buffer[i * 3] =
              static_cast<unsigned char>(256 * intensity.clamp(r_val));
          scanline_buffer[i * 3 + 1] =
              static_cast<unsigned char>(256 * intensity.clamp(g_val));
          scanline_buffer[i * 3 + 2] =
              static_cast<unsigned char>(256 * intensity.clamp(b_val));
          row_completed_pixels++;
        }

        if (row_completed_pixels > 0) {
          {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            int idx = j * image_width * 3;
            std::copy(scanline_buffer.begin(),
                      scanline_buffer.begin() + row_completed_pixels * 3,
                      buffer.begin() + idx);
          }

          int completed = completed_pixels.fetch_add(row_completed_pixels) +
                          row_completed_pixels;
          progress.store(static_cast<float>(completed) / total_pixels);

          int since_update =
              pixels_since_update.fetch_add(row_completed_pixels) +
              row_completed_pixels;

          if (since_update >= update_frequency || completed == total_pixels) {
            texture_needs_update.store(true);
            pixels_since_update.store(0);

            static std::atomic<int> last_reported_percent{-1};
            int current_percent = static_cast<int>(
                static_cast<float>(completed) / total_pixels * 100);
            if (current_percent >= last_reported_percent.load() + 10) {
              std::cout << "Progress: " << current_percent << "%" << std::endl;
              last_reported_percent.store(current_percent);
            }
          }
        }

        if (j % std::max(1, image_height / 20) == 0) {
          texture_needs_update.store(true);
        }
      }
    };

    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(worker);
    }

    for (auto &t : threads) {
      if (t.joinable()) {
        t.join();
      }
    }

    if (!should_stop.load()) {
      std::cout << "Render completed: " << completed_pixels.load() << " pixels"
                << std::endl;
    } else {
      std::cout << "Render stopped at " << completed_pixels.load() << "/"
                << total_pixels << " pixels" << std::endl;
    }
  }

  // Original render method for compatibility
  void render_to_buffer(const hittable &world,
                        std::vector<unsigned char> &buffer) {
    std::mutex dummy_mutex;
    std::atomic<float> dummy_progress{0.0f};
    std::atomic<bool> dummy_stop{false};
    std::atomic<bool> dummy_texture_update{false};
    render_to_buffer_with_progress(world, buffer, dummy_mutex, dummy_progress,
                                   dummy_stop, dummy_texture_update);
  }

private:
  int image_height;
  point3 center;
  point3 pixel00_loc;
  vec3 pixel_delta_u;
  vec3 pixel_delta_v;
  vec3 u, v, w;
  vec3 defocus_disk_u;
  vec3 defocus_disk_v;

  void initialize() {
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    center = lookfrom;

    // Camera setup calculations
    auto theta = degrees_to_radians(vfov);
    auto h = std::tan(theta / 2);
    auto viewport_height = 2 * h * focus_dist;
    auto viewport_width =
        viewport_height * (double(image_width) / image_height);

    // Calculate camera basis vectors
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // Calculate viewport vectors
    auto viewport_u = viewport_width * u;
    auto viewport_v = viewport_height * -v;

    // Calculate pixel delta vectors
    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;

    // Calculate location of upper left pixel
    auto viewport_upper_left =
        center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Calculate defocus disk basis vectors
    auto defocus_radius =
        focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
  }

  ray get_ray(int i, int j) const {
    // Get a randomly sampled camera ray for pixel (i,j)
    vec3 offset = enable_antialiasing ? sample_square() : vec3(0, 0, 0);
    auto pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) +
                        ((j + offset.y()) * pixel_delta_v);

    auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
    auto ray_direction = pixel_sample - ray_origin;
    auto ray_time = random_double();

    return ray(ray_origin, ray_direction, ray_time);
  }

  vec3 sample_square() const {
    // Returns a random point in the square surrounding a pixel at the origin
    return vec3(random_double() - 0.5, random_double() - 0.5, 0);
  }

  point3 defocus_disk_sample() const {
    // Returns a random point in the camera defocus disk
    auto p = random_in_unit_disk();
    return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
  }

  color ray_color(const ray &r, int depth, const hittable &world) const {
    if (depth <= 0) {
      return color(0, 0, 0);
    }

    hit_record rec;

    if (!world.hit(r, interval(0.001, infinity), rec)) {
      return background;
    }
    ray scattered;
    color attenuation;

    // Unconditionally grab any light being emitted by the material we hit.
    // If it's not a light, this safely returns color(0,0,0).
    color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);
    if (enable_shadows && rec.mat->scatter(r, rec, attenuation, scattered)) {
      color color_from_scatter;
      if (enable_reflections || enable_refractions) {
        color_from_scatter =
            attenuation * ray_color(scattered, depth - 1, world);
      } else {
        vec3 light_dir = unit_vector(vec3(1, 1, 1));
        double light_intensity = std::max(0.0, dot(rec.normal, light_dir));
        color_from_scatter =
            attenuation * (0.3 + 0.7 * light_intensity); // Ambient + diffuse
      }
      // Return BOTH the emitted light + the scattered light!
      // Usually one of these is 0, but both are included for mathematical
      // completeness
      return color_from_emission + color_from_scatter;
    }

    // If it failed to scatter (e.g., it is a pure Light element, or shadows
    // are off), it MUST return the emitted light
    return color_from_emission;
  }
};

#endif // !CAMERA_HPP
