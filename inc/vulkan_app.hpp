#ifndef VULKAN_APP_HPP
#define VULKAN_APP_HPP

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "camera.hpp"
#include "cuda_structs.hpp"
#include "hittable_list.hpp"

const int kMaxFramesInFlight = 2;
const std::vector<const char*> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME};

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  bool is_complete() { return graphics_family.has_value() && present_family.has_value(); }
};

struct SwapchainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

enum class Scenes { STATIC, MOTION, CHECKERED, EARTH, PERLIN, QUAD, LIGHT, CORNELL, SMOKE, FINAL, CUSTOM };

// Single source of truth for CLI scene names, so --scene and any future scene
// listing cannot disagree about what exists.
struct SceneName {
  const char* name;
  Scenes scene;
};
inline constexpr SceneName kSceneNames[] = {
    {"static", Scenes::STATIC},    {"motion", Scenes::MOTION}, {"checkered", Scenes::CHECKERED},
    {"earth", Scenes::EARTH},      {"perlin", Scenes::PERLIN}, {"quad", Scenes::QUAD},
    {"light", Scenes::LIGHT},      {"cornell", Scenes::CORNELL}, {"smoke", Scenes::SMOKE},
    {"final", Scenes::FINAL},      {"custom", Scenes::CUSTOM},
};

inline bool scene_from_name(const std::string& s, Scenes& out) {
  for (const auto& e : kSceneNames)
    if (s == e.name) {
      out = e.scene;
      return true;
    }
  return false;
}

inline const char* name_from_scene(Scenes s) {
  for (const auto& e : kSceneNames)
    if (e.scene == s) return e.name;
  return "?";
}

inline std::string scene_name_list() {
  std::string r;
  for (const auto& e : kSceneNames) {
    if (!r.empty()) r += ", ";
    r += e.name;
  }
  return r;
}

// Headless render parameters. Also seeds the GUI's initial values so both
// entry points honour the same flags.
struct HeadlessOptions {
  enum class Backend { GPU, CPU, BOTH };

  bool enabled = false;
  Scenes scene = Scenes::STATIC;
  int width = 800;
  int spp = 10;
  int max_depth = 10;
  Backend backend = Backend::GPU;
  std::string out = "render"; // "<out>_gpu.ppm" / "<out>_cpu.ppm"
};

class VulkanApp {
public:
  explicit VulkanApp(const HeadlessOptions& opts = {});
  ~VulkanApp();

  void run();
  void run_headless();

private:
  HeadlessOptions opts_;
  bool headless_;
  GLFWwindow* window_ = nullptr;

  // Vulkan Core
  VkInstance instance_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  VkQueue present_queue_ = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> swapchain_images_;
  std::vector<VkImageView> swapchain_image_views_;
  VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
  VkExtent2D swapchain_extent_{};

  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline graphics_pipeline_ = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> framebuffers_;

  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> command_buffers_;

  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

  // Sync
  std::vector<VkSemaphore> image_available_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> in_flight_fences_;
  uint32_t current_frame_ = 0;

  // Interop & Viewport State
  VkImage interop_image_ = VK_NULL_HANDLE;
  VkBuffer interop_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory interop_memory_ = VK_NULL_HANDLE;
  VkImageView interop_view_ = VK_NULL_HANDLE;
  VkSampler interop_sampler_ = VK_NULL_HANDLE;
  VkDescriptorSet imgui_texture_ = VK_NULL_HANDLE;
  void* cuda_interop_pointer_ = nullptr;

  int current_width_ = 800;
  int current_height_ = 450;
  float aspect_ratio_ = 16.0f / 9.0f;

  // Raytracer Params
  hittable_list world_;
  camera cam_;
  int samples_per_pixel_ = 10;
  int max_depth_ = 10;
  float background_color_[3] = {0.70f, 0.80f, 1.00f};
  float camera_pos_[3] = {13, 2, 3};
  float camera_target_[3] = {0, 0, 0};
  float camera_fov_ = 20.0f;
  float defocus_angle_ = 0.6f;
  float focus_distance_ = 10.0f;
  int image_width_ = 800;
  Scenes scene_type_ = Scenes::STATIC;

  // GPU Data
  std::vector<LinearBVHNode> gpu_bvh_nodes_;
  std::vector<PrimitiveGPU> gpu_primitives_;
  std::vector<MaterialGPU> gpu_materials_;
  std::vector<TextureGPU> gpu_textures_;
  std::vector<PerlinDataGPU> gpu_perlin_;
  std::vector<unsigned char> gpu_image_buffer_;

  // CPU Render Bridge
  // Linear HDR, 3 floats/pixel. Pre-tonemap and pre-gamma: this is the float64
  // reference image the validation harness compares against the device.
  std::vector<float> cpu_render_buffer_;
  std::mutex cpu_buffer_mutex_;
  std::thread cpu_render_thread_;

  // UI/Render Control
  std::atomic<bool> is_rendering_{false};
  std::atomic<float> render_progress_{0.0f};
  std::atomic<bool> should_stop_render_{false};
  std::atomic<bool> texture_needs_update_{false};
  bool trigger_render_ = false;
  bool use_gpu_render_ = true;
  float render_time_ = 0.0f;

  void setup_world();
  void setup_camera();

  // Vulkan Internal
  void init_window();
  void init_vulkan();
  void main_loop();
  void draw_frame();
  void cleanup();

  void init_instance();
  void create_surface();
  void pick_physical_device();
  void create_logical_device();
  void create_swapchain();
  void create_image_views();
  void create_render_pass();
  void create_descriptor_set_layout();
  void create_graphics_pipeline();
  void create_framebuffers();
  void create_command_pool();
  void create_command_buffers();
  void create_sync_objects();
  void create_interop_image();
  void create_descriptors();
  void setup_cuda_interop();
  void setup_imgui();

  uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties);
  QueueFamilyIndices find_queue_families(VkPhysicalDevice device);
  bool is_device_suitable(VkPhysicalDevice device);
  bool check_device_extension_support(VkPhysicalDevice device);
  SwapchainSupportDetails query_swapchain_support(VkPhysicalDevice device);
  VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& availableFormats);
  VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& availablePresentModes);
  VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities);
  std::vector<const char*> get_required_extensions();
  std::vector<char> read_spirv(const std::string& filename);
  VkShaderModule create_shader_module(const std::vector<char>& code);
  int export_memory_fd();
  void transition_image_layout(VkCommandBuffer cb, VkImage image, VkFormat format, VkImageLayout oldLayout,
                               VkImageLayout newLayout);

  static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                       VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                       void* pUserData);
  void setup_debug_messenger();
  bool check_validation_layer_support();
  void export_gpu_ppm(const std::string& path);
  void export_cpu_ppm(const std::string& path);
};

#endif
