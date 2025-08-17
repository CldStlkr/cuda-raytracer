#ifndef CAMERA_HPP
#define CAMERA_HPP
#include "color.hpp"
#include "hittable.hpp"
#include "material.hpp"
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

class camera {
public:
  double aspect_ratio = 1.0;
  int image_width = 100;
  int samples_per_pixel = 10;
  int max_depth = 10;
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
  void render_to_buffer_with_progress(const hittable& world,
                                      std::vector<unsigned char>& buffer,
                                      std::mutex& buffer_mutex,
                                      std::atomic<float>& progress,
                                      const std::atomic<bool>& should_stop,
                                      std::atomic<bool>& texture_needs_update) {
    initialize();

    // Ensure buffer is properly sized
    {
      std::lock_guard<std::mutex> lock(buffer_mutex);
      buffer.resize(image_width * image_height * 3);
      std::fill(buffer.begin(), buffer.end(), 0); // Clear to black
    }

    int total_pixels = image_width * image_height;
    int completed_pixels = 0;
    int pixels_since_update = 0;
    const int update_frequency =
        std::max(1, total_pixels / 100); // Update every 1% of pixels

    auto last_update_time = std::chrono::steady_clock::now();
    const auto update_interval =
        std::chrono::milliseconds(100); // Update at most every 100ms

    std::cout << "Rendering " << image_width << "x" << image_height << " with "
              << samples_per_pixel << " samples per pixel..." << std::endl;

    for (int j = 0; j < image_height && !should_stop.load(); j++) {
      for (int i = 0; i < image_width && !should_stop.load(); i++) {
        color pixel_color(0, 0, 0);
        int sample_count = enable_antialiasing ? samples_per_pixel : 1;

        // Render all samples for this pixel
        for (int sample = 0; sample < sample_count && !should_stop.load();
             sample++) {
          ray r = get_ray(i, j);
          pixel_color += ray_color(r, max_depth, world);
        }

        if (should_stop.load()) break;

        // Convert pixel color to RGB bytes
        double scale = 1.0 / sample_count;
        pixel_color *= scale;

        auto r_val = linear_to_gamma(pixel_color.x());
        auto g_val = linear_to_gamma(pixel_color.y());
        auto b_val = linear_to_gamma(pixel_color.z());

        static const interval intensity(0.000, 0.999);
        unsigned char r_byte =
            static_cast<unsigned char>(256 * intensity.clamp(r_val));
        unsigned char g_byte =
            static_cast<unsigned char>(256 * intensity.clamp(g_val));
        unsigned char b_byte =
            static_cast<unsigned char>(256 * intensity.clamp(b_val));

        // Update buffer (thread-safe)
        {
          std::lock_guard<std::mutex> lock(buffer_mutex);
          int idx = (j * image_width + i) * 3;
          buffer[idx] = r_byte;
          buffer[idx + 1] = g_byte;
          buffer[idx + 2] = b_byte;
        }

        completed_pixels++;
        pixels_since_update++;

        // Update progress
        float current_progress =
            static_cast<float>(completed_pixels) / total_pixels;
        progress.store(current_progress);

        // Trigger texture update periodically for real-time display
        auto now = std::chrono::steady_clock::now();
        bool time_for_update = (now - last_update_time) >= update_interval;
        bool enough_pixels = pixels_since_update >= update_frequency;

        if (time_for_update || enough_pixels ||
            completed_pixels == total_pixels) {
          texture_needs_update.store(true);
          pixels_since_update = 0;
          last_update_time = now;

          // Optional: print progress every 10%
          static int last_reported_percent = -1;
          int current_percent = static_cast<int>(current_progress * 100);
          if (current_percent >= last_reported_percent + 10) {
            std::cout << "Progress: " << current_percent << "%" << std::endl;
            last_reported_percent = current_percent;
          }
        }
      }

      // Update texture at the end of each row for progressive display
      if (j % std::max(1, image_height / 20) == 0) { // Update every 5% of rows
        texture_needs_update.store(true);
      }
    }

    if (!should_stop.load()) {
      std::cout << "Render completed: " << completed_pixels << " pixels"
                << std::endl;
    } else {
      std::cout << "Render stopped at " << completed_pixels << "/"
                << total_pixels << " pixels" << std::endl;
    }
  }

  // Original render method for compatibility
  void render_to_buffer(const hittable& world,
                        std::vector<unsigned char>& buffer) {
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

    return ray(ray_origin, ray_direction);
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

  color ray_color(const ray& r, int depth, const hittable& world) const {
    if (depth <= 0) {
      return color(0, 0, 0);
    }

    hit_record rec;
    if (world.hit(r, interval(0.001, infinity), rec)) {
      ray scattered;
      color attenuation;

      // Handle material scattering based on enabled features
      if (enable_shadows && rec.mat->scatter(r, rec, attenuation, scattered)) {
        // Only do recursive ray tracing if reflections/refractions are enabled
        if (enable_reflections || enable_refractions) {
          return attenuation * ray_color(scattered, depth - 1, world);
        } else {
          // Simplified shading without recursion
          vec3 light_dir =
              unit_vector(vec3(1, 1, 1)); // Simple directional light
          double light_intensity = std::max(0.0, dot(rec.normal, light_dir));
          return attenuation *
                 (0.3 + 0.7 * light_intensity); // Ambient + diffuse
        }
      }
      return color(0, 0, 0);
    }

    // Background gradient (sky)
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
  }
};

#endif // !CAMERA_HPP
