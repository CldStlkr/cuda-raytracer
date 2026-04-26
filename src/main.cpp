#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Include ray tracing headers
#include "bvh.hpp"
#include "camera.hpp"
#include "color.hpp"
#include "constant_medium.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"
#include "interval.hpp"
#include "material.hpp"
#include "quad.hpp"
#include "ray.hpp"
#include "rt.hpp"
#include "sphere.hpp"
#include "texture.hpp"
#include "vec3.hpp"

#include "cuda/vec.cuh"
#include "cuda_structs.hpp"

using std::atomic;
using std::make_shared;
using std::mutex;
using std::thread;
using std::vector;

enum Scenes {
  STATIC,
  MOTION_BLUR,
  CHECKERED_SPHERES,
  EARTH,
  PERLIN,
  QUAD,
  SIMPLE_LIGHT,
  CORNELL_BOX,
  CORNELL_SMOKE,
  FINAL_SCENE,
  CUSTOM_SHOWCASE,
};

extern "C" void launch_render(RenderConfig config, BVHBuffer bvh,
                              PrimitiveBuffer prims, MaterialBuffer mats,
                              TextureBuffer texs, PerlinBuffer perlin,
                              ImageArrayBuffer images);

void render_with_cuda(vector<unsigned char> &buffer, int width, int height,
                      int samples, int max_depth,
                      const vector<LinearBVHNode> &nodes,
                      const vector<PrimitiveGPU> &prims,
                      const vector<MaterialGPU> &mats,
                      const vector<TextureGPU> &texs,
                      const vector<PerlinDataGPU> &perlin,
                      const vector<unsigned char> &images, const camera &cam) {
  printf("CPU: Starting CUDA render %dx%d\n", width, height);

  // Use CUDA GPU type to match kernel expectations
  vector<vec3_gpu> frame_buffer(width * height);

  // Initialize frame buffer to avoid garbage data
  for (auto &pixel : frame_buffer) {
    pixel = vec3_gpu(0.0f, 0.0f, 0.0f);
  }

  printf("CPU: Allocated frame buffer of size %zu\n", frame_buffer.size());

  RenderConfig config;
  config.frame_buffer = frame_buffer.data();
  config.width = width;
  config.height = height;
  config.samples_per_pixel = samples;
  config.max_depth = max_depth;
  config.background =
      Vec3f{float(cam.background.x()), float(cam.background.y()),
            float(cam.background.z())};

  config.lookfrom = Vec3f{float(cam.lookfrom.x()), float(cam.lookfrom.y()),
                          float(cam.lookfrom.z())};
  config.lookat = Vec3f{float(cam.lookat.x()), float(cam.lookat.y()),
                        float(cam.lookat.z())};
  config.vup =
      Vec3f{float(cam.vup.x()), float(cam.vup.y()), float(cam.vup.z())};
  config.vfov = cam.vfov;
  config.defocus_angle = cam.defocus_angle;
  config.focus_dist = cam.focus_dist;

  BVHBuffer bvh = {nodes.data(), nodes.size()};
  PrimitiveBuffer p_buf = {prims.data(), prims.size()};
  MaterialBuffer m_buf = {mats.data(), mats.size()};
  TextureBuffer t_buf = {texs.data(), texs.size()};
  PerlinBuffer per_buf = {perlin.data(), perlin.size()};
  ImageArrayBuffer i_buf = {images.data(), images.size()};

  // Call CUDA kernel
  launch_render(config, bvh, p_buf, m_buf, t_buf, per_buf, i_buf);

  printf("CPU: CUDA render returned, converting to RGB buffer\n");

  // Convert vec3_gpu frame_buffer (0–1) → unsigned char (0–255)
  buffer.resize(width * height * 3);

  for (size_t i = 0; i < frame_buffer.size(); i++) {
    // Clamp values to [0, 1] range before conversion
    // Gamma 2.0 Correction via sqrt() before clamping
    float r = fminf(fmaxf(sqrtf(frame_buffer[i].x()), 0.0f), 1.0f);
    float g = fminf(fmaxf(sqrtf(frame_buffer[i].y()), 0.0f), 1.0f);
    float b = fminf(fmaxf(sqrtf(frame_buffer[i].z()), 0.0f), 1.0f);

    buffer[i * 3 + 0] = static_cast<unsigned char>(255.99f * r);
    buffer[i * 3 + 1] = static_cast<unsigned char>(255.99f * g);
    buffer[i * 3 + 2] = static_cast<unsigned char>(255.99f * b);
  }

  printf("CPU: Conversion complete\n");
}

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

class RayTracerApp {
private:
  GLFWwindow *window;
  GLuint texture_id;

  // Ray tracing objects
  camera cam;
  hittable_list world;

  std::vector<LinearBVHNode> gpu_bvh_nodes;
  std::vector<PrimitiveGPU> gpu_primitives;
  std::vector<MaterialGPU> gpu_materials;
  std::vector<TextureGPU> gpu_textures;
  std::vector<PerlinDataGPU> gpu_perlin;
  std::vector<unsigned char> gpu_image_buffer;

  // Rendering state
  vector<unsigned char> image_buffer;
  mutex buffer_mutex;
  atomic<bool> is_rendering{false};
  atomic<bool> should_stop_render{false};
  atomic<bool> texture_needs_update{false};
  atomic<bool> use_gpu_render{true};
  thread render_thread;

  // Image dimensions
  atomic<int> current_image_width{400};
  atomic<int> current_image_height{225};

  // GUI state
  bool show_controls = true;
  bool show_debug = false;
  atomic<float> render_progress{0.0f};
  Scenes s = Scenes::STATIC; // 0 = Static, 1 = Moving Spheres

  // Camera parameters for UI
  float camera_pos[3] = {13.0f, 2.0f, 3.0f};
  float camera_target[3] = {0.0f, 0.0f, 0.0f};
  float camera_fov = 20.0f;
  float focus_distance = 10.0f;
  float defocus_angle = 0.6f;
  float aspect_ratio = 16.0f / 9.0f;
  float background_color[3] = {0.70f, 0.80f, 1.00f};
  int image_width = 400;
  int samples_per_pixel = 10;
  int max_depth = 10;

  // Timing
  std::chrono::steady_clock::time_point render_start_time;
  atomic<float> render_time_seconds{0.0f};

public:
  RayTracerApp() : window(nullptr), texture_id(0) {
    setup_world();
    setup_camera();
  }

  ~RayTracerApp() { cleanup_render_thread(); }

  void cleanup_render_thread() {
    if (render_thread.joinable()) {
      should_stop_render.store(true);
      render_thread.join();
    }
  }

  bool initialize() {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {
      std::cerr << "Failed to initialize GLFW" << std::endl;
      return false;
    }

    // GL 3.3 + GLSL 330
    const char *glsl_version = "#version 330 core";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(1600, 1000, "Ray Tracer - Real-time GUI", nullptr,
                              nullptr);
    if (!window) {
      std::cerr << "Failed to create GLFW window" << std::endl;
      glfwTerminate();
      return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    if (!gladLoadGL(glfwGetProcAddress)) {
      std::cerr << "Failed to initialize GLAD" << std::endl;
      return false;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Create OpenGL texture
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Initialize with a blank texture
    update_texture_size();

    return true;
  }

  void setup_world() {
    world.clear();

    // Default book 1 camera matching majority of scenes
    camera_pos[0] = 13.0f;
    camera_pos[1] = 2.0f;
    camera_pos[2] = 3.0f;
    camera_target[0] = 0.0f;
    camera_target[1] = 0.0f;
    camera_target[2] = 0.0f;
    camera_fov = 20.0f;
    aspect_ratio = 16.0f / 9.0f;
    background_color[0] = 0.70f;
    background_color[1] = 0.80f;
    background_color[2] = 1.00f;
    defocus_angle = 0.6f;
    focus_distance = 10.0f;

    if (s == Scenes::STATIC) {
      auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
      world.add(
          make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

      for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
          auto choose_mat = random_double();
          point3 center(a + 0.9 * random_double(), 0.2,
                        b + 0.9 * random_double());

          if ((center - point3(4, 0.2, 0)).length() > 0.9) {
            shared_ptr<material> sphere_material;

            if (choose_mat < 0.8) {
              // diffuse
              auto albedo = color::random() * color::random();
              sphere_material = make_shared<lambertian>(albedo);
              world.add(make_shared<sphere>(center, 0.2, sphere_material));
            } else if (choose_mat < 0.95) {
              // metal
              auto albedo = color::random(0.5, 1);
              auto fuzz = random_double(0, 0.5);
              sphere_material = make_shared<metal>(albedo, fuzz);
              world.add(make_shared<sphere>(center, 0.2, sphere_material));
            } else {
              // glass
              sphere_material = make_shared<dielectric>(1.5);
              world.add(make_shared<sphere>(center, 0.2, sphere_material));
            }
          }
        }
      }

      auto material1 = make_shared<dielectric>(1.5);
      world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

      auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
      world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

      auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
      world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));
    } else if (s == Scenes::CORNELL_SMOKE) {
      camera_pos[0] = 278.0f;
      camera_pos[1] = 278.0f;
      camera_pos[2] = -800.0f;
      camera_target[0] = 278.0f;
      camera_target[1] = 278.0f;
      camera_target[2] = 0.0f;
      camera_fov = 40.0f;
      aspect_ratio = 1.0f;
      background_color[0] = 0.0f;
      background_color[1] = 0.0f;
      background_color[2] = 0.0f;

      auto red = make_shared<lambertian>(color(.65, .05, .05));
      auto white = make_shared<lambertian>(color(.73, .73, .73));
      auto green = make_shared<lambertian>(color(.12, .45, .15));
      auto light = make_shared<diffuse_light>(color(7, 7, 7));

      world.add(make_shared<quad>(point3(555, 0, 0), vec3(0, 555, 0),
                                  vec3(0, 0, 555), green));
      world.add(make_shared<quad>(point3(0, 0, 0), vec3(0, 555, 0),
                                  vec3(0, 0, 555), red));
      world.add(make_shared<quad>(point3(113, 554, 127), vec3(330, 0, 0),
                                  vec3(0, 0, 305), light));
      world.add(make_shared<quad>(point3(0, 555, 0), vec3(555, 0, 0),
                                  vec3(0, 0, 555), white));
      world.add(make_shared<quad>(point3(0, 0, 0), vec3(555, 0, 0),
                                  vec3(0, 0, 555), white));
      world.add(make_shared<quad>(point3(0, 0, 555), vec3(555, 0, 0),
                                  vec3(0, 555, 0), white));

      shared_ptr<hittable> box1 =
          box(point3(0, 0, 0), point3(165, 330, 165), white);
      box1 = make_shared<rotate_y>(box1, 15);
      box1 = make_shared<translate>(box1, vec3(265, 0, 295));

      shared_ptr<hittable> box2 =
          box(point3(0, 0, 0), point3(165, 165, 165), white);
      box2 = make_shared<rotate_y>(box2, -18);
      box2 = make_shared<translate>(box2, vec3(130, 0, 65));

      world.add(make_shared<constant_medium>(box1, 0.01, color(0, 0, 0)));
      world.add(make_shared<constant_medium>(box2, 0.01, color(1, 1, 1)));

      camera cam;

    } else if (s == Scenes::CORNELL_BOX) {
      camera_pos[0] = 278.0f;
      camera_pos[1] = 278.0f;
      camera_pos[2] = -800.0f;
      camera_target[0] = 278.0f;
      camera_target[1] = 278.0f;
      camera_target[2] = 0.0f;
      camera_fov = 40.0f;
      aspect_ratio = 1.0f;
      background_color[0] = 0.0f;
      background_color[1] = 0.0f;
      background_color[2] = 0.0f;

      auto red = make_shared<lambertian>(color(.65, .05, .05));
      auto white = make_shared<lambertian>(color(.73, .73, .73));
      auto green = make_shared<lambertian>(color(.12, .45, .15));
      auto light = make_shared<diffuse_light>(color(15, 15, 15));

      world.add(make_shared<quad>(point3(555, 0, 0), vec3(0, 555, 0),
                                  vec3(0, 0, 555), green));
      world.add(make_shared<quad>(point3(0, 0, 0), vec3(0, 555, 0),
                                  vec3(0, 0, 555), red));
      world.add(make_shared<quad>(point3(343, 554, 332), vec3(-130, 0, 0),
                                  vec3(0, 0, -105), light));
      world.add(make_shared<quad>(point3(0, 0, 0), vec3(555, 0, 0),
                                  vec3(0, 0, 555), white));
      world.add(make_shared<quad>(point3(555, 555, 555), vec3(-555, 0, 0),
                                  vec3(0, 0, -555), white));
      world.add(make_shared<quad>(point3(0, 0, 555), vec3(555, 0, 0),
                                  vec3(0, 555, 0), white));

      shared_ptr<hittable> box1 =
          box(point3(0, 0, 0), point3(165, 330, 165), white);
      box1 = make_shared<rotate_y>(box1, 15);
      box1 = std::make_shared<translate>(box1, vec3(265, 0, 295));
      world.add(box1);

      shared_ptr<hittable> box2 =
          box(point3(0, 0, 0), point3(165, 165, 165), white);
      box2 = make_shared<rotate_y>(box2, -18);
      box2 = make_shared<translate>(box2, vec3(130, 0, 65));
      world.add(box2);

    } else if (s == Scenes::SIMPLE_LIGHT) {
      camera_pos[0] = 26.0f;
      camera_pos[1] = 3.0f;
      camera_pos[2] = 6.0f;
      camera_target[0] = 0.0f;
      camera_target[1] = 2.0f;
      camera_target[2] = 0.0f;
      background_color[0] = 0.0f;
      background_color[1] = 0.0f;
      background_color[2] = 0.0f;

      auto pertext = make_shared<noise_texture>(4);
      world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                    make_shared<lambertian>(pertext)));
      world.add(make_shared<sphere>(point3(0, 2, 0), 2,
                                    make_shared<lambertian>(pertext)));

      auto difflight = make_shared<diffuse_light>(color(4, 4, 4));
      world.add(make_shared<sphere>(point3(0, 7, 0), 2, difflight));
      world.add(make_shared<quad>(point3(3, 1, -2), vec3(2, 0, 0),
                                  vec3(0, 2, 0), difflight));

    } else if (s == Scenes::QUAD) {
      camera_pos[0] = 0.0f;
      camera_pos[1] = 0.0f;
      camera_pos[2] = 9.0f;
      camera_fov = 80.0f;
      aspect_ratio = 1.0f;

      // Materials
      auto left_red = make_shared<lambertian>(color(1.0, 0.2, 0.2));
      auto back_green = make_shared<lambertian>(color(0.2, 1.0, 0.2));
      auto right_blue = make_shared<lambertian>(color(0.2, 0.2, 1.0));
      auto upper_orange = make_shared<lambertian>(color(1.0, 0.5, 0.0));
      auto lower_teal = make_shared<lambertian>(color(0.2, 0.8, 0.8));

      // Quads
      world.add(make_shared<quad>(point3(-3, -2, 5), vec3(0, 0, -4),
                                  vec3(0, 4, 0), left_red));
      world.add(make_shared<quad>(point3(-2, -2, 0), vec3(4, 0, 0),
                                  vec3(0, 4, 0), back_green));
      world.add(make_shared<quad>(point3(3, -2, 1), vec3(0, 0, 4),
                                  vec3(0, 4, 0), right_blue));
      world.add(make_shared<quad>(point3(-2, 3, 1), vec3(4, 0, 0),
                                  vec3(0, 0, 4), upper_orange));
      world.add(make_shared<quad>(point3(-2, -3, 5), vec3(4, 0, 0),
                                  vec3(0, 0, -4), lower_teal));

    } else if (s == Scenes::PERLIN) {

      auto pertext = make_shared<noise_texture>(5);
      world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                    make_shared<lambertian>(pertext)));
      world.add(make_shared<sphere>(point3(0, 2, 0), 2,
                                    make_shared<lambertian>(pertext)));

    } else if (s == Scenes::EARTH) {
      auto earth_texture = make_shared<image_texture>("earthmap.jpg");
      auto earth_surface = make_shared<lambertian>(earth_texture);
      auto globe = make_shared<sphere>(point3(0, 0, 0), 2, earth_surface);

      world.add(globe);
    } else if (s == Scenes::CHECKERED_SPHERES) {

      auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1),
                                                  color(.9, .9, .9));

      world.add(make_shared<sphere>(point3(0, -10, 0), 10,
                                    make_shared<lambertian>(checker)));
      world.add(make_shared<sphere>(point3(0, 10, 0), 10,
                                    make_shared<lambertian>(checker)));
    } else if (s == Scenes::FINAL_SCENE) {
      camera_pos[0] = 478.0f;
      camera_pos[1] = 278.0f;
      camera_pos[2] = -600.0f;
      camera_target[0] = 278.0f;
      camera_target[1] = 278.0f;
      camera_target[2] = 0.0f;
      camera_fov = 40.0f;
      aspect_ratio = 1.0f;
      background_color[0] = 0.0f;
      background_color[1] = 0.0f;
      background_color[2] = 0.0f;
      defocus_angle = 0.0f;

      hittable_list boxes1;
      auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

      int boxes_per_side = 20;
      for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
          auto w = 100.0;
          auto x0 = -1000.0 + i * w;
          auto z0 = -1000.0 + j * w;
          auto y0 = 0.0;
          auto x1 = x0 + w;
          auto y1 = random_double(1, 101);
          auto z1 = z0 + w;

          boxes1.add(box(point3(x0, y0, z0), point3(x1, y1, z1), ground));
        }
      }

      world.add(make_shared<bvh_node>(boxes1));

      auto light = make_shared<diffuse_light>(color(7, 7, 7));
      world.add(make_shared<quad>(point3(123, 554, 147), vec3(300, 0, 0),
                                  vec3(0, 0, 265), light));

      auto center1 = point3(400, 400, 200);
      auto center2 = center1 + vec3(30, 0, 0);
      auto sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
      world.add(make_shared<sphere>(center1, center2, 50, sphere_material));

      world.add(make_shared<sphere>(point3(260, 150, 45), 50,
                                    make_shared<dielectric>(1.5)));
      world.add(
          make_shared<sphere>(point3(0, 150, 145), 50,
                              make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)));

      auto boundary = make_shared<sphere>(point3(360, 150, 145), 70,
                                          make_shared<dielectric>(1.5));
      world.add(boundary);
      world.add(
          make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
      boundary = make_shared<sphere>(point3(0, 0, 0), 5000,
                                     make_shared<dielectric>(1.5));
      world.add(make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));

      auto emat =
          make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
      world.add(make_shared<sphere>(point3(400, 200, 400), 100, emat));
      auto pertext = make_shared<noise_texture>(0.2);
      world.add(make_shared<sphere>(point3(220, 280, 300), 80,
                                    make_shared<lambertian>(pertext)));

      hittable_list boxes2;
      auto white = make_shared<lambertian>(color(.73, .73, .73));
      int ns = 1000;
      for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<sphere>(point3::random(0, 165), 10, white));
      }

      world.add(make_shared<translate>(
          make_shared<rotate_y>(make_shared<bvh_node>(boxes2), 15),
          vec3(-100, 270, 395)));

    } else if (s == Scenes::CUSTOM_SHOWCASE) {
      camera_pos[0] = 13.0f;
      camera_pos[1] = 2.0f;
      camera_pos[2] = 3.0f;
      camera_target[0] = 0.0f;
      camera_target[1] = 0.0f;
      camera_target[2] = 0.0f;
      camera_fov = 20.0f;
      background_color[0] = 0.0f;
      background_color[1] = 0.0f;
      background_color[2] = 0.0f;
      aspect_ratio = 1.0f;
      defocus_angle = 0.0f;
      focus_distance = 10.0f;

      // Floor
      auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1),
                                                  color(.9, .9, .9));
      world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                    make_shared<lambertian>(checker)));

      // Central Pillar (Perlin)
      auto pertext = make_shared<noise_texture>(1.5);
      shared_ptr<hittable> central_pillar = box(
          point3(-1, 0, -1), point3(1, 1, 1), make_shared<lambertian>(pertext));
      world.add(central_pillar);

      // The Prism (Volume)
      shared_ptr<hittable> prism_boundary = box(
          point3(-1, 1.1, -1), point3(1, 2.5, 1), make_shared<dielectric>(1.5));
      world.add(make_shared<constant_medium>(prism_boundary, 0.1,
                                             color(0.8, 0.8, 1.0)));

      // Earth Orbiter
      auto earth_texture = make_shared<image_texture>("earthmap.jpg");
      auto earth_surface = make_shared<lambertian>(earth_texture);
      world.add(make_shared<sphere>(point3(0, 1.8, 0), 0.4, earth_surface));

      // Internal Glow
      auto light_mat = make_shared<diffuse_light>(color(10, 10, 10));
      world.add(make_shared<sphere>(point3(0, 1.8, 0), 0.1, light_mat));

      // Rotating Metallic Corner Pillars
      auto metal_mat = make_shared<metal>(color(0.8, 0.8, 0.8), 0.0);
      for (int i = 0; i < 4; i++) {
        float angle = i * 90.0f;
        float rad = angle * 3.14159 / 180.0;
        float x = 4.0 * cos(rad);
        float z = 4.0 * sin(rad);

        shared_ptr<hittable> corner_pillar =
            box(point3(-0.5, 0, -0.5), point3(0.5, 3.0, 0.5), metal_mat);
        corner_pillar = make_shared<rotate_y>(corner_pillar, angle + 45);
        corner_pillar = make_shared<translate>(corner_pillar, vec3(x, 0, z));
        world.add(corner_pillar);
      }

      // Motion Blurred Projectiles
      auto blur_mat = make_shared<lambertian>(color(0.7, 0.3, 0.1));
      for (int i = 0; i < 5; i++) {
        point3 center1(-5, 1 + i, 5 - i * 2);
        point3 center2(-3, 1 + i, 5 - i * 2);
        world.add(make_shared<sphere>(center1, center2, 0.2, blur_mat));
      }

      // Ceiling Light
      auto ceiling_light = make_shared<diffuse_light>(color(4, 4, 4));
      world.add(make_shared<quad>(point3(-5, 10, -5), vec3(10, 0, 0),
                                  vec3(0, 0, 10), ceiling_light));

    } else {
      // Book 2 Moving Spheres
      auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
      world.add(
          make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

      auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1),
                                                  color(.9, .9, .9));
      world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                    make_shared<lambertian>(checker)));

      for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
          auto choose_mat = random_double();
          point3 center(a + 0.9 * random_double(), 0.2,
                        b + 0.9 * random_double());

          if ((center - point3(4, 0.2, 0)).length() > 0.9) {
            shared_ptr<material> sphere_material;

            if (choose_mat < 0.8) {
              // diffuse
              auto albedo = color::random() * color::random();
              sphere_material = make_shared<lambertian>(albedo);
              auto center2 = center + vec3(0, random_double(0, .5), 0);
              world.add(
                  make_shared<sphere>(center, center2, 0.2, sphere_material));
            } else if (choose_mat < 0.95) {
              // metal
              auto albedo = color::random(0.5, 1);
              auto fuzz = random_double(0, 0.5);
              sphere_material = make_shared<metal>(albedo, fuzz);
              world.add(make_shared<sphere>(center, 0.2, sphere_material));
            } else {
              // glass
              sphere_material = make_shared<dielectric>(1.5);
              world.add(make_shared<sphere>(center, 0.2, sphere_material));
            }
          }
        }
      }

      auto material1 = make_shared<dielectric>(1.5);
      world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

      auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
      world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

      auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
      world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

      world = hittable_list(make_shared<bvh_node>(world));

      camera cam;
    }

    gpu_bvh_nodes.clear();
    gpu_primitives.clear();
    gpu_materials.clear();
    gpu_textures.clear();
    gpu_perlin.clear();
    gpu_image_buffer.clear();

    std::unordered_map<material *, int> temporary_material_map;
    std::unordered_map<texture *, int> temporary_texture_map;
    auto root_bvh = std::make_shared<bvh_node>(world);
    flatten_hittable(root_bvh, gpu_bvh_nodes, gpu_primitives, gpu_materials,
                     gpu_textures, gpu_perlin, gpu_image_buffer,
                     temporary_material_map, temporary_texture_map);
  }

  void setup_camera() {
    cam.aspect_ratio = aspect_ratio;
    cam.image_width = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth = max_depth;
    cam.background =
        color(background_color[0], background_color[1], background_color[2]);
    cam.vfov = camera_fov;
    cam.lookfrom = point3(camera_pos[0], camera_pos[1], camera_pos[2]);
    cam.lookat = point3(camera_target[0], camera_target[1], camera_target[2]);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = defocus_angle;
    cam.focus_dist = focus_distance;

    // Update current dimensions
    int new_height = int(image_width / cam.aspect_ratio);
    new_height = (new_height < 1) ? 1 : new_height;
    current_image_width.store(image_width);
    current_image_height.store(new_height);
  }

  void update_texture_size() {
    int width = current_image_width.load();
    int height = current_image_height.load();

    std::lock_guard<mutex> lock(buffer_mutex);
    image_buffer.resize(width * height * 3, 0);

    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, image_buffer.data());
  }

  void start_render() {
    if (is_rendering.load())
      return;

    // Update camera settings
    setup_camera();
    update_texture_size();

    cleanup_render_thread();

    should_stop_render.store(false);
    is_rendering.store(true);
    render_progress.store(0.0f);
    render_time_seconds.store(0.0f);
    texture_needs_update.store(false);

    render_start_time = std::chrono::steady_clock::now();

    // Forward-declared from cuda_renderer.cu

    render_thread = thread([this]() {
      try {
        std::cout << "Starting render: " << current_image_width.load() << "x"
                  << current_image_height.load() << " with "
                  << cam.samples_per_pixel << " samples" << std::endl;

        if (use_gpu_render.load()) {
          // CUDA Render Path
          render_with_cuda(image_buffer, current_image_width.load(),
                           current_image_height.load(), cam.samples_per_pixel,
                           cam.max_depth, gpu_bvh_nodes, gpu_primitives,
                           gpu_materials, gpu_textures, gpu_perlin,
                           gpu_image_buffer, cam);
        } else {
          // CPU Render Path
          cam.render_to_buffer_with_progress(
              world, image_buffer, buffer_mutex, render_progress,
              should_stop_render, texture_needs_update);
        }

        if (!should_stop_render.load()) {
          render_progress.store(1.0f);
          texture_needs_update.store(true);
          std::cout << "Render completed!" << std::endl;
        } else {
          std::cout << "Render stopped by user" << std::endl;
        }
      } catch (const std::exception &e) {
        std::cerr << "Render error: " << e.what() << std::endl;
      }
      is_rendering.store(false);
    });
  }

  void update_texture() {
    if (texture_needs_update.load()) {
      std::lock_guard<mutex> lock(buffer_mutex);

      int width = current_image_width.load();
      int height = current_image_height.load();

      if (!image_buffer.empty() &&
          image_buffer.size() >= (size_t)(width * height * 3)) {
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB,
                        GL_UNSIGNED_BYTE, image_buffer.data());
        texture_needs_update.store(false);
      }
    }
  }

  void export_ppm() {
    std::lock_guard<mutex> lock(buffer_mutex);

    if (image_buffer.empty()) {
      std::cout << "No image to export!" << std::endl;
      return;
    }

    std::ofstream file("output.ppm");
    if (!file.is_open()) {
      std::cerr << "Failed to open output.ppm for writing" << std::endl;
      return;
    }

    int width = current_image_width.load();
    int height = current_image_height.load();

    file << "P3\n" << width << " " << height << "\n255\n";

    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i++) {
        int idx = (j * width + i) * 3;
        file << static_cast<int>(image_buffer[idx]) << " "
             << static_cast<int>(image_buffer[idx + 1]) << " "
             << static_cast<int>(image_buffer[idx + 2]) << "\n";
      }
    }

    file.close();
    std::cout << "Image exported to output.ppm" << std::endl;
  }

  void render_gui() {
    // Update render timing
    if (is_rendering.load()) {
      auto now = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - render_start_time);
      render_time_seconds.store(duration.count() / 1000.0f);
    }

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Control panel
    if (show_controls) {
      ImGui::Begin("Ray Tracer Controls", &show_controls,
                   ImGuiWindowFlags_AlwaysAutoResize);

      // Rendering controls
      if (ImGui::CollapsingHeader("Rendering Options",
                                  ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Anti-aliasing", &cam.enable_antialiasing);
        ImGui::Checkbox("Shadows", &cam.enable_shadows);
        ImGui::Checkbox("Reflections", &cam.enable_reflections);
        ImGui::Checkbox("Refractions", &cam.enable_refractions);

        ImGui::Separator();

        bool is_gpu = use_gpu_render.load();
        if (ImGui::Checkbox("Use GPU Acceleration (CUDA)", &is_gpu)) {
          use_gpu_render.store(is_gpu);
        }

        ImGui::Separator();

        bool params_changed = false;

        // Scene selection
        const char *scenes[] = {"Static Spheres",    "Moving Spheres (Book 2)",
                                "Checkered Spheres", "Earth",
                                "Perlin Sphere",     "Quads",
                                "Simple Light",      "Cornell Box",
                                "Cornell Smoke",     "Final Scene",
                                "Obsidian Prism"};
        int scene_idx = static_cast<int>(s);
        if (ImGui::Combo("Scene", &scene_idx, scenes, IM_ARRAYSIZE(scenes))) {
          s = static_cast<Scenes>(scene_idx);
          setup_world();
          params_changed = true;
        }

        params_changed |=
            ImGui::ColorEdit3("Background Color", background_color);
        params_changed |=
            ImGui::SliderInt("Samples per pixel", &samples_per_pixel, 1, 10000);
        params_changed |= ImGui::SliderInt("Max depth", &max_depth, 1, 50);
        params_changed |=
            ImGui::SliderInt("Image width", &image_width, 100, 1600);

        int estimated_height = int(image_width / 16.0 * 9.0);
        ImGui::Text("Image size: %dx%d (%d pixels)", image_width,
                    estimated_height, image_width * estimated_height);

        if (params_changed && !is_rendering.load()) {
          setup_camera();
          update_texture_size();
        }
      }

      // Camera controls
      if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("FOV", &camera_fov, 10.0f, 120.0f);
        ImGui::SliderFloat("Focus distance", &focus_distance, 0.1f, 20.0f);
        ImGui::SliderFloat("Defocus angle", &defocus_angle, 0.0f, 10.0f);
        ImGui::SliderFloat3("Position", camera_pos, -20.0f, 20.0f);
        ImGui::SliderFloat3("Target", camera_target, -20.0f, 20.0f);
      }

      ImGui::Separator();

      // Render controls
      if (is_rendering.load()) {
        float progress = render_progress.load();
        float time_elapsed = render_time_seconds.load();

        ImGui::Text("Rendering... (%.1fs)", time_elapsed);
        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f),
                           (std::to_string(int(progress * 100)) + "%").c_str());

        if (progress > 0.01f) {
          float estimated_total = time_elapsed / progress;
          float remaining = estimated_total - time_elapsed;
          ImGui::Text("Estimated remaining: %.1fs", remaining);
        }

        if (ImGui::Button("Stop Render", ImVec2(-1.0f, 0.0f))) {
          should_stop_render.store(true);
        }
      } else {
        if (ImGui::Button("Start Render", ImVec2(-1.0f, 0.0f))) {
          start_render();
        }

        ImGui::Separator();

        if (ImGui::Button("Export PPM", ImVec2(-1.0f, 0.0f)) &&
            !image_buffer.empty()) {
          export_ppm();
        }
      }

      // Debug info
      ImGui::Separator();
      ImGui::Checkbox("Show Debug Info", &show_debug);

      if (show_debug) {
        ImGui::Text("Texture ID: %u", texture_id);
        ImGui::Text("Buffer size: %zu bytes", image_buffer.size());
        ImGui::Text("Update pending: %s",
                    texture_needs_update.load() ? "Yes" : "No");

        ImGuiIO &io = ImGui::GetIO();
        ImGui::Text("FPS: %.1f", io.Framerate);
      }

      ImGui::End();
    }

    // Image display window
    ImGui::Begin("Rendered Image", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    if (texture_id) {
      int width = current_image_width.load();
      int height = current_image_height.load();

      // Calculate display size to fit nicely in the window
      float max_display_width = 800.0f;
      float max_display_height = 600.0f;

      float aspect_ratio =
          static_cast<float>(width) / static_cast<float>(height);
      float display_width =
          std::min(max_display_width, static_cast<float>(width));
      float display_height = display_width / aspect_ratio;

      if (display_height > max_display_height) {
        display_height = max_display_height;
        display_width = display_height * aspect_ratio;
      }

      ImGui::Text("Image: %dx%d (Display: %.0fx%.0f)", width, height,
                  display_width, display_height);

      if (is_rendering.load()) {
        float progress = render_progress.load();
        ImGui::Text("Rendering Progress: %.1f%%", progress * 100.0f);
      } else if (!image_buffer.empty()) {
        ImGui::Text("Render complete! Time: %.1fs", render_time_seconds.load());
      }

      // Display the image
      ImGui::Image(reinterpret_cast<void *>(static_cast<intptr_t>(texture_id)),
                   ImVec2(display_width, display_height), ImVec2(0, 0),
                   ImVec2(1, 1), ImVec4(1, 1, 1, 1), ImVec4(0, 0, 0, 0));
    } else {
      ImGui::Text("No image rendered yet.");
      ImGui::Text("Click 'Start Render' to generate an image.");
    }

    ImGui::End();

    // Menu bar
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("Show Controls", nullptr, &show_controls);
        ImGui::MenuItem("Show Debug", nullptr, &show_debug);
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Export PPM", "Ctrl+E", false,
                            !image_buffer.empty())) {
          export_ppm();
        }
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Render")) {
        if (ImGui::MenuItem("Start", "Ctrl+R", false, !is_rendering.load())) {
          start_render();
        }
        if (ImGui::MenuItem("Stop", "Esc", false, is_rendering.load())) {
          should_stop_render.store(true);
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }

    // Rendering
    ImGui::Render();
  }

  void run() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      // Update texture frequently for real-time display
      update_texture();

      // Render GUI
      render_gui();

      // OpenGL rendering
      int display_w, display_h;
      glfwGetFramebufferSize(window, &display_w, &display_h);
      glViewport(0, 0, display_w, display_h);
      glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(window);
    }
  }

  void cleanup() {
    cleanup_render_thread();

    if (texture_id) {
      glDeleteTextures(1, &texture_id);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (window) {
      glfwDestroyWindow(window);
    }
    glfwTerminate();
  }
};

int main() {
  RayTracerApp app;

  if (!app.initialize()) {
    std::cerr << "Failed to initialize application" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Ray Tracer started successfully!" << std::endl;
  std::cout << "Use the controls panel to adjust settings and start rendering."
            << std::endl;
  std::cout << "The rendered image will appear in real-time in the GUI!"
            << std::endl;

  app.run();
  app.cleanup();

  std::cout << "Application closed cleanly." << std::endl;
  return EXIT_SUCCESS;
}
