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
#include "camera.hpp"
#include "color.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"
#include "interval.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "rt.hpp"
#include "sphere.hpp"
#include "vec3.hpp"

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

class RayTracerApp {
private:
  GLFWwindow* window;
  GLuint texture_id;

  // Ray tracing objects
  camera cam;
  hittable_list world;

  // Rendering state
  std::vector<unsigned char> image_buffer;
  std::mutex buffer_mutex;
  std::atomic<bool> is_rendering{false};
  std::atomic<bool> should_stop_render{false};
  std::atomic<bool> texture_needs_update{false};
  std::thread render_thread;

  // Image dimensions
  std::atomic<int> current_image_width{400};
  std::atomic<int> current_image_height{225};

  // GUI state
  bool show_controls = true;
  bool show_debug = false;
  std::atomic<float> render_progress{0.0f};

  // Camera parameters for UI
  float camera_pos[3] = {13.0f, 2.0f, 3.0f};
  float camera_target[3] = {0.0f, 0.0f, 0.0f};
  float camera_fov = 20.0f;
  float focus_distance = 10.0f;
  float defocus_angle = 0.6f;
  int image_width = 400;
  int samples_per_pixel = 10;
  int max_depth = 10;

  // Timing
  std::chrono::steady_clock::time_point render_start_time;
  std::atomic<float> render_time_seconds{0.0f};

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
    const char* glsl_version = "#version 330 core";
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
    ImGuiIO& io = ImGui::GetIO();
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

    // Ground sphere
    auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(
        std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    // Central large glass sphere
    auto glass_material = std::make_shared<dielectric>(1.5);
    world.add(std::make_shared<sphere>(point3(0, 1, 0), 1.0, glass_material));

    // Surrounding spheres in a circle pattern
    auto red_diffuse = std::make_shared<lambertian>(color(0.7, 0.2, 0.2));
    world.add(std::make_shared<sphere>(point3(-5, 1, 0), 1.0, red_diffuse));

    auto blue_diffuse = std::make_shared<lambertian>(color(0.2, 0.2, 0.7));
    world.add(std::make_shared<sphere>(point3(5, 1, 0), 1.0, blue_diffuse));

    auto green_diffuse = std::make_shared<lambertian>(color(0.2, 0.7, 0.2));
    world.add(std::make_shared<sphere>(point3(0, 1, -5), 1.0, green_diffuse));

    auto yellow_diffuse = std::make_shared<lambertian>(color(0.7, 0.7, 0.2));
    world.add(std::make_shared<sphere>(point3(0, 1, 5), 1.0, yellow_diffuse));

    // Metal spheres at diagonal positions
    auto gold_metal = std::make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);
    world.add(std::make_shared<sphere>(point3(-3.5, 1, -3.5), 1.0, gold_metal));

    auto silver_metal = std::make_shared<metal>(color(0.8, 0.8, 0.9), 0.1);
    world.add(std::make_shared<sphere>(point3(3.5, 1, 3.5), 1.0, silver_metal));

    auto copper_metal = std::make_shared<metal>(color(0.7, 0.4, 0.3), 0.2);
    world.add(
        std::make_shared<sphere>(point3(-3.5, 1, 3.5), 1.0, copper_metal));

    auto chrome_metal = std::make_shared<metal>(color(0.9, 0.9, 0.9), 0.0);
    world.add(
        std::make_shared<sphere>(point3(3.5, 1, -3.5), 1.0, chrome_metal));

    // Smaller spheres at different heights
    auto purple_diffuse = std::make_shared<lambertian>(color(0.6, 0.2, 0.6));
    world.add(
        std::make_shared<sphere>(point3(-2, 0.5, -2), 0.5, purple_diffuse));

    auto orange_diffuse = std::make_shared<lambertian>(color(0.8, 0.4, 0.1));
    world.add(std::make_shared<sphere>(point3(2, 0.5, 2), 0.5, orange_diffuse));

    auto cyan_diffuse = std::make_shared<lambertian>(color(0.2, 0.6, 0.6));
    world.add(std::make_shared<sphere>(point3(-2, 0.5, 2), 0.5, cyan_diffuse));

    auto pink_diffuse = std::make_shared<lambertian>(color(0.8, 0.4, 0.6));
    world.add(std::make_shared<sphere>(point3(2, 0.5, -2), 0.5, pink_diffuse));

    // Some elevated spheres for depth
    auto white_diffuse = std::make_shared<lambertian>(color(0.9, 0.9, 0.9));
    world.add(std::make_shared<sphere>(point3(-1, 2, -1), 0.3, white_diffuse));

    auto black_diffuse = std::make_shared<lambertian>(color(0.1, 0.1, 0.1));
    world.add(std::make_shared<sphere>(point3(1, 2, 1), 0.3, black_diffuse));

    // Glass spheres at different positions
    auto glass2 = std::make_shared<dielectric>(1.3);
    world.add(std::make_shared<sphere>(point3(-6, 0.7, -2), 0.7, glass2));

    auto glass3 = std::make_shared<dielectric>(1.8);
    world.add(std::make_shared<sphere>(point3(6, 0.7, 2), 0.7, glass3));

    // Far background spheres for depth
    auto distant_metal = std::make_shared<metal>(color(0.5, 0.5, 0.7), 0.3);
    world.add(
        std::make_shared<sphere>(point3(-10, 1.5, -8), 1.5, distant_metal));

    auto distant_diffuse = std::make_shared<lambertian>(color(0.4, 0.6, 0.4));
    world.add(
        std::make_shared<sphere>(point3(8, 1.2, -10), 1.2, distant_diffuse));
  }

  void setup_camera() {
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth = max_depth;
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

    std::lock_guard<std::mutex> lock(buffer_mutex);
    image_buffer.resize(width * height * 3, 0);

    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, image_buffer.data());
  }

  void start_render() {
    if (is_rendering.load()) return;

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

    render_thread = std::thread([this]() {
      try {
        std::cout << "Starting render: " << current_image_width.load() << "x"
                  << current_image_height.load() << " with "
                  << cam.samples_per_pixel << " samples" << std::endl;

        // Render with progress tracking and periodic texture updates
        cam.render_to_buffer_with_progress(world, image_buffer, buffer_mutex,
                                           render_progress, should_stop_render,
                                           texture_needs_update);

        if (!should_stop_render.load()) {
          render_progress.store(1.0f);
          texture_needs_update.store(true);
          std::cout << "Render completed!" << std::endl;
        } else {
          std::cout << "Render stopped by user" << std::endl;
        }
      } catch (const std::exception& e) {
        std::cerr << "Render error: " << e.what() << std::endl;
      }
      is_rendering.store(false);
    });
  }

  void update_texture() {
    if (texture_needs_update.load()) {
      std::lock_guard<std::mutex> lock(buffer_mutex);

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
    std::lock_guard<std::mutex> lock(buffer_mutex);

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

        bool params_changed = false;
        params_changed |=
            ImGui::SliderInt("Samples per pixel", &samples_per_pixel, 1, 500);
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

        ImGuiIO& io = ImGui::GetIO();
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
      ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture_id)),
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
