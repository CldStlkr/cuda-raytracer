#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Validation layers — enabled in Debug builds only
// ---------------------------------------------------------------------------
#ifdef NDEBUG
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// Device extensions we require on the chosen GPU
const std::vector<const char*> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
};

// ---------------------------------------------------------------------------
// Small helper structs
// ---------------------------------------------------------------------------

// Which queue families does our chosen physical device support?
// optional<> means "not yet found" vs "found at index N"
struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> present_family;

    bool is_complete() const {
        return graphics_family.has_value() && present_family.has_value();
    }
};

// What surface formats / present modes does the swapchain support?
struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   present_modes;
};

// ---------------------------------------------------------------------------
// VulkanApp
// ---------------------------------------------------------------------------
class VulkanApp {
public:
    VulkanApp();
    ~VulkanApp();

    // Entry point — call this from main()
    void run();

private:
    // ── Window ───────────────────────────────────────────────────────────────
    GLFWwindow* window_ = nullptr;
    static constexpr int kWidth  = 1600;
    static constexpr int kHeight = 1000;

    void init_window();

    // ── Core Vulkan handles (Phase 2a) ───────────────────────────────────────
    VkInstance               instance_        = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR             surface_         = VK_NULL_HANDLE;
    VkPhysicalDevice         physical_device_ = VK_NULL_HANDLE;
    VkDevice                 device_          = VK_NULL_HANDLE;
    VkQueue                  graphics_queue_  = VK_NULL_HANDLE;
    VkQueue                  present_queue_   = VK_NULL_HANDLE;

    void init_instance();
    void setup_debug_messenger();
    void create_surface();
    void pick_physical_device();
    void create_logical_device();

    // ── Swapchain (Phase 2b) ─────────────────────────────────────────────────
    VkSwapchainKHR             swapchain_            = VK_NULL_HANDLE;
    std::vector<VkImage>       swapchain_images_;
    std::vector<VkImageView>   swapchain_image_views_;
    VkFormat                   swapchain_format_     = VK_FORMAT_UNDEFINED;
    VkExtent2D                 swapchain_extent_     = {};

    void create_swapchain();
    void create_image_views();

    // ── Render pass + pipeline (Phase 2c) ────────────────────────────────────
    VkRenderPass               render_pass_          = VK_NULL_HANDLE;
    VkDescriptorSetLayout      descriptor_set_layout_= VK_NULL_HANDLE;
    VkPipelineLayout           pipeline_layout_      = VK_NULL_HANDLE;
    VkPipeline                 graphics_pipeline_    = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;

    void create_render_pass();
    void create_descriptor_set_layout();
    void create_graphics_pipeline();
    void create_framebuffers();

    // ── Commands + sync (Phase 2d) ───────────────────────────────────────────
    VkCommandPool                command_pool_   = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;

    // Per-frame sync primitives (double-buffered)
    static constexpr int kMaxFramesInFlight = 2;
    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence>     in_flight_fences_;
    uint32_t                 current_frame_ = 0;

    void create_command_pool();
    void create_command_buffers();
    void create_sync_objects();

    // ── CUDA interop (Phase 3) ───────────────────────────────────────────────
    // These will be filled in later — declared now so the struct is stable
    VkImage        interop_image_  = VK_NULL_HANDLE;
    VkDeviceMemory interop_memory_ = VK_NULL_HANDLE;
    VkImageView    interop_view_   = VK_NULL_HANDLE;
    VkSampler      interop_sampler_= VK_NULL_HANDLE;

    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet  descriptor_set_  = VK_NULL_HANDLE;

    // ── Frame loop ───────────────────────────────────────────────────────────
    void init_vulkan();
    void main_loop();
    void draw_frame();
    void cleanup();

    // ── Helpers ──────────────────────────────────────────────────────────────
    bool                   check_validation_layer_support();
    std::vector<const char*> get_required_extensions();
    bool                   is_device_suitable(VkPhysicalDevice device);
    bool                   check_device_extension_support(VkPhysicalDevice device);
    QueueFamilyIndices     find_queue_families(VkPhysicalDevice device);
    SwapchainSupportDetails query_swapchain_support(VkPhysicalDevice device);
    VkSurfaceFormatKHR     choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& formats);
    VkPresentModeKHR       choose_swap_present_mode(const std::vector<VkPresentModeKHR>& modes);
    VkExtent2D             choose_swap_extent(const VkSurfaceCapabilitiesKHR& caps);
    std::vector<char>      read_spirv(const std::string& path);
    VkShaderModule         create_shader_module(const std::vector<char>& code);

    // Debug messenger callback — must be static (C linkage requirement)
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
        VkDebugUtilsMessageTypeFlagsEXT             type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void*                                       user_data);
};
