#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "bvh.hpp"
#include "constant_medium.hpp"
#include "material.hpp"
#include "quad.hpp"
#include "rt.hpp"
#include "sphere.hpp"
#include "texture.hpp"
#include "vulkan_app.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include <cuda/std/span>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <unistd.h>

using cuda::std::span;

extern "C" void launch_render(RenderConfig config, span<LinearBVHNode> h_bvh, span<PrimitiveGPU> h_prims,
                              span<MaterialGPU> h_mats, span<TextureGPU> h_texs, span<PerlinDataGPU> h_perlin,
                              span<unsigned char> h_images);

extern "C" void* import_vulkan_memory(int fd, size_t size);
extern "C" void cleanup_cuda_interop();

int flatten_hittable(std::shared_ptr<hittable> root, std::vector<LinearBVHNode>& nodes,
                     std::vector<PrimitiveGPU>& primitives, std::vector<MaterialGPU>& materials,
                     std::vector<TextureGPU>& textures, std::vector<PerlinDataGPU>& perlin,
                     std::vector<unsigned char>& images, std::unordered_map<material*, int>& mat_map,
                     std::unordered_map<texture*, int>& tex_map);

uint32_t VulkanApp::find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

VulkanApp::VulkanApp(bool headless) : headless_(headless) {
  if (!headless_) {
    init_window();
    init_vulkan();
  }
  setup_world();
  setup_camera();
}

VulkanApp::~VulkanApp() { cleanup(); }

void VulkanApp::init_window() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, VK_FALSE);
  window_ = glfwCreateWindow(1600, 1000, "CUDA-Vulkan Raytracer Studio", nullptr, nullptr);
}

void VulkanApp::init_vulkan() {
  init_instance();
  create_surface();
  pick_physical_device();
  create_logical_device();
  create_swapchain();
  create_image_views();
  create_render_pass();
  create_descriptor_set_layout();
  create_graphics_pipeline();
  create_framebuffers();
  create_command_pool();
  create_command_buffers();
  create_interop_image();
  create_descriptors();
  setup_cuda_interop();
  setup_imgui();
  create_sync_objects();
}

void VulkanApp::run() {
  if (headless_) {
    run_headless();
  } else {
    main_loop();
  }
}
void VulkanApp::main_loop() {
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
    draw_frame();
  }
  vkDeviceWaitIdle(device_);
}

void VulkanApp::draw_frame() {
  vkWaitForFences(device_, 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);
  uint32_t imageIndex;
  VkResult result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, image_available_semaphores_[current_frame_],
                                          VK_NULL_HANDLE, &imageIndex);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) return;
  vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);

  VkCommandBuffer cb = command_buffers_[current_frame_];

  int expected_height = (int)(image_width_ / aspect_ratio_);
  if ((image_width_ != current_width_ || expected_height != current_height_) && !is_rendering_) {
    vkDeviceWaitIdle(device_);
    cudaDeviceSynchronize();
    current_width_ = image_width_;
    current_height_ = (int)(current_width_ / aspect_ratio_);

    cleanup_cuda_interop();

    if (imgui_texture_) {
      ImGui_ImplVulkan_RemoveTexture(imgui_texture_);
      imgui_texture_ = VK_NULL_HANDLE;
    }
    vkDestroyImageView(device_, interop_view_, nullptr);
    vkDestroySampler(device_, interop_sampler_, nullptr);
    vkDestroyImage(device_, interop_image_, nullptr);
    vkDestroyBuffer(device_, interop_buffer_, nullptr);
    vkFreeMemory(device_, interop_memory_, nullptr);

    create_interop_image();
    setup_cuda_interop();
    imgui_texture_ =
        ImGui_ImplVulkan_AddTexture(interop_sampler_, interop_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    setup_camera();
  }

  if (trigger_render_ && !is_rendering_) {
    is_rendering_ = true;
    render_progress_ = 0.0f;
    trigger_render_ = false;

    if (use_gpu_render_) {
      auto start = std::chrono::high_resolution_clock::now();

      RenderConfig config;
      config.frame_buffer = (vec3_gpu*)cuda_interop_pointer_;
      config.width = current_width_;
      config.height = current_height_;
      config.samples_per_pixel = samples_per_pixel_;
      config.max_depth = max_depth_;
      config.background = Vec3f{background_color_[0], background_color_[1], background_color_[2]};
      config.lookfrom = Vec3f{camera_pos_[0], camera_pos_[1], camera_pos_[2]};
      config.lookat = Vec3f{camera_target_[0], camera_target_[1], camera_target_[2]};
      config.vup = Vec3f{0, 1, 0};
      config.vfov = camera_fov_;
      config.defocus_angle = defocus_angle_;
      config.focus_dist = focus_distance_;

      span<LinearBVHNode> bvh = {gpu_bvh_nodes_.data(), gpu_bvh_nodes_.size()};
      span<PrimitiveGPU> p_buf = {gpu_primitives_.data(), gpu_primitives_.size()};
      span<MaterialGPU> m_buf = {gpu_materials_.data(), gpu_materials_.size()};
      span<TextureGPU> t_buf = {gpu_textures_.data(), gpu_textures_.size()};
      span<PerlinDataGPU> per_buf = {gpu_perlin_.data(), gpu_perlin_.size()};
      span<unsigned char> i_buf = {gpu_image_buffer_.data(), gpu_image_buffer_.size()};

      if (cuda_interop_pointer_) {
        launch_render(config, bvh, p_buf, m_buf, t_buf, per_buf, i_buf);
        cudaDeviceSynchronize();
      }

      vkResetCommandBuffer(cb, 0);
      VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      vkBeginCommandBuffer(cb, &bi);
      transition_image_layout(cb, interop_image_, VK_FORMAT_R32G32B32A32_SFLOAT,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      VkBufferImageCopy region = {.bufferOffset = 0,
                                  .bufferRowLength = 0,
                                  .bufferImageHeight = 0,
                                  .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                       .mipLevel = 0,
                                                       .baseArrayLayer = 0,
                                                       .layerCount = 1},
                                  .imageOffset = {0, 0, 0},
                                  .imageExtent = {(uint32_t)current_width_, (uint32_t)current_height_, 1}};
      vkCmdCopyBufferToImage(cb, interop_buffer_, interop_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
      transition_image_layout(cb, interop_image_, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      vkEndCommandBuffer(cb);
      VkSubmitInfo ts = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
      vkQueueSubmit(graphics_queue_, 1, &ts, VK_NULL_HANDLE);
      vkQueueWaitIdle(graphics_queue_);

      auto end = std::chrono::high_resolution_clock::now();
      render_time_ = std::chrono::duration<float>(end - start).count();
      is_rendering_ = false;
    } else {
      if (cpu_render_thread_.joinable()) cpu_render_thread_.join();
      auto start = std::chrono::high_resolution_clock::now();
      cpu_render_thread_ = std::thread([this, start]() {
        setup_camera();
        cam_.render_to_buffer_with_progress(world_, cpu_render_buffer_, cpu_buffer_mutex_, render_progress_,
                                            should_stop_render_, texture_needs_update_);
        auto end = std::chrono::high_resolution_clock::now();
        render_time_ = std::chrono::duration<float>(end - start).count();
        is_rendering_ = false;
      });
    }
  }

  if (texture_needs_update_) {
    texture_needs_update_ = false;
    std::vector<float> float_buffer(current_width_ * current_height_ * 4);
    {
      std::lock_guard<std::mutex> lock(cpu_buffer_mutex_);
      if (cpu_render_buffer_.size() >= (size_t)current_width_ * current_height_ * 3) {
        for (int i = 0; i < current_width_ * current_height_; i++) {
          float_buffer[i * 4 + 0] = cpu_render_buffer_[i * 3 + 0] / 255.0f;
          float_buffer[i * 4 + 1] = cpu_render_buffer_[i * 3 + 1] / 255.0f;
          float_buffer[i * 4 + 2] = cpu_render_buffer_[i * 3 + 2] / 255.0f;
          float_buffer[i * 4 + 3] = 1.0f;
        }
      }
    }
    if (cuda_interop_pointer_) {
      cudaMemcpy(cuda_interop_pointer_, float_buffer.data(), float_buffer.size() * sizeof(float),
                 cudaMemcpyHostToDevice);
      vkResetCommandBuffer(cb, 0);
      VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      vkBeginCommandBuffer(cb, &bi);
      transition_image_layout(cb, interop_image_, VK_FORMAT_R32G32B32A32_SFLOAT,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      VkBufferImageCopy region = {.bufferOffset = 0,
                                  .bufferRowLength = 0,
                                  .bufferImageHeight = 0,
                                  .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                       .mipLevel = 0,
                                                       .baseArrayLayer = 0,
                                                       .layerCount = 1},
                                  .imageOffset = {0, 0, 0},
                                  .imageExtent = {(uint32_t)current_width_, (uint32_t)current_height_, 1}};
      vkCmdCopyBufferToImage(cb, interop_buffer_, interop_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
      transition_image_layout(cb, interop_image_, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      vkEndCommandBuffer(cb);
      VkSubmitInfo ts = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
      vkQueueSubmit(graphics_queue_, 1, &ts, VK_NULL_HANDLE);
      vkQueueWaitIdle(graphics_queue_);
    }
  }

  vkResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);

  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Raytracer Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  if (ImGui::CollapsingHeader("Rendering Options", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Use GPU Acceleration (CUDA)", &use_gpu_render_);
    const char* scenes[] = {"Static", "Motion Blur", "Checkered",     "Earth",       "Perlin",         "Quad",
                            "Light",  "Cornell Box", "Cornell Smoke", "Final Scene", "Custom Showcase"};
    int s_idx = (int)scene_type_;
    if (ImGui::Combo("Scene", &s_idx, scenes, IM_ARRAYSIZE(scenes))) {
      scene_type_ = (Scenes)s_idx;
      setup_world();
    }
    ImGui::ColorEdit3("Background", background_color_);
    ImGui::SliderInt("Samples", &samples_per_pixel_, 1, 10000);
    ImGui::SliderInt("Max Depth", &max_depth_, 1, 50);
    ImGui::SliderInt("Image Width", &image_width_, 100, 1600);
  }
  if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("FOV", &camera_fov_, 10.0f, 120.0f);
    ImGui::SliderFloat("Focus Dist", &focus_distance_, 0.1f, 50.0f);
    ImGui::SliderFloat("Defocus", &defocus_angle_, 0.0f, 10.0f);
    ImGui::SliderFloat3("Position", camera_pos_, -20.0f, 20.0f);
    ImGui::SliderFloat3("Target", camera_target_, -20.0f, 20.0f);
  }
  ImGui::Separator();
  if (is_rendering_) {
    float p = render_progress_.load();
    ImGui::Text("Rendering... (%.1f%%)", p * 100.0f);
    ImGui::ProgressBar(p, ImVec2(-1.0f, 0.0f));
    if (ImGui::Button("STOP RENDER", ImVec2(-1.0f, 30.0f))) {
      should_stop_render_ = true;
    }
  } else {
    if (ImGui::Button("START RENDER", ImVec2(-1.0f, 40.0f))) {
      trigger_render_ = true;
    }
    if (render_time_ > 0) ImGui::Text("Last Render Time: %.3fs", render_time_);
  }
  ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
  ImGui::End();

  ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_FirstUseEver);
  ImGui::Begin("Viewport", nullptr);
  ImVec2 view_size = ImGui::GetContentRegionAvail();
  if (view_size.x > 0 && view_size.y > 0 && imgui_texture_) {
    float img_aspect = (float)current_width_ / (float)current_height_;
    float display_w = view_size.x;
    float display_h = display_w / img_aspect;
    if (display_h > view_size.y) {
      display_h = view_size.y;
      display_w = display_h * img_aspect;
    }
    ImGui::Image((ImTextureID)imgui_texture_, ImVec2(display_w, display_h));
  }
  ImGui::End();

  VkClearValue cv = {.color = {{0.1f, 0.1f, 0.11f, 1.0f}}};
  VkRenderPassBeginInfo rp = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                              .renderPass = render_pass_,
                              .framebuffer = framebuffers_[imageIndex],
                              .renderArea = {.offset = {0, 0}, .extent = swapchain_extent_},
                              .clearValueCount = 1,
                              .pClearValues = &cv};
  vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);
  ImGui::Render();
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
  vkCmdEndRenderPass(cb);
  vkEndCommandBuffer(cb);

  VkSemaphore ws[] = {image_available_semaphores_[current_frame_]};
  VkPipelineStageFlags wst[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSemaphore ss[] = {render_finished_semaphores_[current_frame_]};
  VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                     .waitSemaphoreCount = 1,
                     .pWaitSemaphores = ws,
                     .pWaitDstStageMask = wst,
                     .commandBufferCount = 1,
                     .pCommandBuffers = &cb,
                     .signalSemaphoreCount = 1,
                     .pSignalSemaphores = ss};
  vkQueueSubmit(graphics_queue_, 1, &si, in_flight_fences_[current_frame_]);
  VkPresentInfoKHR present_info = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                   .waitSemaphoreCount = 1,
                                   .pWaitSemaphores = ss,
                                   .swapchainCount = 1,
                                   .pSwapchains = &swapchain_,
                                   .pImageIndices = &imageIndex};
  vkQueuePresentKHR(present_queue_, &present_info);
  current_frame_ = (current_frame_ + 1) % kMaxFramesInFlight;
}

void VulkanApp::init_instance() {
  VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                          .pApplicationName = "Raytracer Studio",
                          .applicationVersion = 0,
                          .pEngineName = "None",
                          .engineVersion = 0,
                          .apiVersion = VK_API_VERSION_1_2};
  auto ex = get_required_extensions();
  VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                             .pApplicationInfo = &ai,
                             .enabledExtensionCount = (uint32_t)ex.size(),
                             .ppEnabledExtensionNames = ex.data()};
  vkCreateInstance(&ci, nullptr, &instance_);
}

void VulkanApp::create_surface() { glfwCreateWindowSurface(instance_, window_, nullptr, &surface_); }

void VulkanApp::pick_physical_device() {
  uint32_t c = 0;
  vkEnumeratePhysicalDevices(instance_, &c, nullptr);
  std::vector<VkPhysicalDevice> d(c);
  vkEnumeratePhysicalDevices(instance_, &c, d.data());
  for (auto dev : d) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(dev, &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && is_device_suitable(dev)) {
      physical_device_ = dev;
      return;
    }
  }
  physical_device_ = d[0];
}

void VulkanApp::create_logical_device() {
  auto i = find_queue_families(physical_device_);
  float p = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> qis;
  std::set<uint32_t> unique = {i.graphics_family.value(), i.present_family.value()};
  for (uint32_t f : unique) {
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = f,
                                  .queueCount = 1,
                                  .pQueuePriorities = &p};
    qis.push_back(qi);
  }
  VkDeviceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .queueCreateInfoCount = (uint32_t)qis.size(),
                           .pQueueCreateInfos = qis.data(),
                           .enabledExtensionCount = (uint32_t)kDeviceExtensions.size(),
                           .ppEnabledExtensionNames = kDeviceExtensions.data()};
  vkCreateDevice(physical_device_, &ci, nullptr, &device_);
  vkGetDeviceQueue(device_, i.graphics_family.value(), 0, &graphics_queue_);
  vkGetDeviceQueue(device_, i.present_family.value(), 0, &present_queue_);
}

void VulkanApp::create_swapchain() {
  auto s = query_swapchain_support(physical_device_);
  auto fmt = choose_swap_surface_format(s.formats);
  auto ext = choose_swap_extent(s.capabilities);
  uint32_t count = s.capabilities.minImageCount + 1;
  if (s.capabilities.maxImageCount > 0 && count > s.capabilities.maxImageCount) count = s.capabilities.maxImageCount;
  VkSwapchainCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                                 .surface = surface_,
                                 .minImageCount = count,
                                 .imageFormat = fmt.format,
                                 .imageColorSpace = fmt.colorSpace,
                                 .imageExtent = ext,
                                 .imageArrayLayers = 1,
                                 .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
  auto i = find_queue_families(physical_device_);
  uint32_t indices[] = {i.graphics_family.value(), i.present_family.value()};
  if (i.graphics_family != i.present_family) {
    ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    ci.queueFamilyIndexCount = 2;
    ci.pQueueFamilyIndices = indices;
  } else {
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }
  ci.preTransform = s.capabilities.currentTransform;
  ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  ci.presentMode = choose_swap_present_mode(s.present_modes);
  ci.clipped = VK_TRUE;
  vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_);
  vkGetSwapchainImagesKHR(device_, swapchain_, &count, nullptr);
  swapchain_images_.resize(count);
  vkGetSwapchainImagesKHR(device_, swapchain_, &count, swapchain_images_.data());
  swapchain_format_ = fmt.format;
  swapchain_extent_ = ext;
}

void VulkanApp::create_image_views() {
  swapchain_image_views_.resize(swapchain_images_.size());
  for (size_t i = 0; i < swapchain_images_.size(); i++) {
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                .image = swapchain_images_[i],
                                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = swapchain_format_,
                                .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                     .baseMipLevel = 0,
                                                     .levelCount = 1,
                                                     .baseArrayLayer = 0,
                                                     .layerCount = 1}};
    vkCreateImageView(device_, &ci, nullptr, &swapchain_image_views_[i]);
  }
}

void VulkanApp::create_render_pass() {
  VkAttachmentDescription ca = {.format = swapchain_format_,
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};
  VkAttachmentReference cr = {.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1, .pColorAttachments = &cr};
  VkRenderPassCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                               .attachmentCount = 1,
                               .pAttachments = &ca,
                               .subpassCount = 1,
                               .pSubpasses = &sub};
  vkCreateRenderPass(device_, &ci, nullptr, &render_pass_);
}

void VulkanApp::create_descriptor_set_layout() {
  VkDescriptorSetLayoutBinding b = {.binding = 0,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    .descriptorCount = 1,
                                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
  VkDescriptorSetLayoutCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b};
  vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descriptor_set_layout_);
}

void VulkanApp::create_graphics_pipeline() {
  VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                   .setLayoutCount = 1,
                                   .pSetLayouts = &descriptor_set_layout_};
  vkCreatePipelineLayout(device_, &li, nullptr, &pipeline_layout_);
}

void VulkanApp::create_framebuffers() {
  framebuffers_.resize(swapchain_image_views_.size());
  for (size_t i = 0; i < swapchain_image_views_.size(); i++) {
    VkImageView a[] = {swapchain_image_views_[i]};
    VkFramebufferCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                  .renderPass = render_pass_,
                                  .attachmentCount = 1,
                                  .pAttachments = a,
                                  .width = swapchain_extent_.width,
                                  .height = swapchain_extent_.height,
                                  .layers = 1};
    vkCreateFramebuffer(device_, &fi, nullptr, &framebuffers_[i]);
  }
}

void VulkanApp::create_command_pool() {
  auto i = find_queue_families(physical_device_);
  VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                .queueFamilyIndex = i.graphics_family.value()};
  vkCreateCommandPool(device_, &pi, nullptr, &command_pool_);
}

void VulkanApp::create_command_buffers() {
  command_buffers_.resize(kMaxFramesInFlight);
  VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                    .commandPool = command_pool_,
                                    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                    .commandBufferCount = (uint32_t)command_buffers_.size()};
  vkAllocateCommandBuffers(device_, &ai, command_buffers_.data());
}

void VulkanApp::create_sync_objects() {
  image_available_semaphores_.resize(kMaxFramesInFlight);
  render_finished_semaphores_.resize(kMaxFramesInFlight);
  in_flight_fences_.resize(kMaxFramesInFlight);
  VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
  for (int i = 0; i < kMaxFramesInFlight; i++) {
    vkCreateSemaphore(device_, &si, nullptr, &image_available_semaphores_[i]);
    vkCreateSemaphore(device_, &si, nullptr, &render_finished_semaphores_[i]);
    vkCreateFence(device_, &fi, nullptr, &in_flight_fences_[i]);
  }
}

void VulkanApp::create_interop_image() {
  VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                          .imageType = VK_IMAGE_TYPE_2D,
                          .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                          .extent = {(uint32_t)current_width_, (uint32_t)current_height_, 1},
                          .mipLevels = 1,
                          .arrayLayers = 1,
                          .samples = VK_SAMPLE_COUNT_1_BIT,
                          .tiling = VK_IMAGE_TILING_OPTIMAL,
                          .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                          .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  vkCreateImage(device_, &ii, nullptr, &interop_image_);

  size_t sz = current_width_ * current_height_ * 4 * sizeof(float);
  VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                         .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .pNext = &eb,
                           .size = sz,
                           .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  vkCreateBuffer(device_, &bi, nullptr, &interop_buffer_);

  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(device_, interop_buffer_, &mr);
  VkMemoryRequirements im_mr;
  vkGetImageMemoryRequirements(device_, interop_image_, &im_mr);

  VkDeviceSize offset = (mr.size + im_mr.alignment - 1) & ~(im_mr.alignment - 1);
  VkDeviceSize total_size = offset + im_mr.size;

  VkExportMemoryAllocateInfo xi = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                   .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  VkMemoryAllocateInfo mi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                             .pNext = &xi,
                             .allocationSize = total_size,
                             .memoryTypeIndex = find_memory_type(mr.memoryTypeBits | im_mr.memoryTypeBits,
                                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  vkAllocateMemory(device_, &mi, nullptr, &interop_memory_);

  vkBindBufferMemory(device_, interop_buffer_, interop_memory_, 0);
  vkBindImageMemory(device_, interop_image_, interop_memory_, offset);

  VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                              .image = interop_image_,
                              .viewType = VK_IMAGE_VIEW_TYPE_2D,
                              .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                              .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                   .baseMipLevel = 0,
                                                   .levelCount = 1,
                                                   .baseArrayLayer = 0,
                                                   .layerCount = 1}};
  vkCreateImageView(device_, &vi, nullptr, &interop_view_);

  VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                            .magFilter = VK_FILTER_LINEAR,
                            .minFilter = VK_FILTER_LINEAR,
                            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                            .mipLodBias = 0.0f,
                            .anisotropyEnable = VK_FALSE,
                            .maxAnisotropy = 1.0f,
                            .compareEnable = VK_FALSE,
                            .compareOp = VK_COMPARE_OP_ALWAYS,
                            .minLod = 0.0f,
                            .maxLod = 1.0f,
                            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                            .unnormalizedCoordinates = VK_FALSE};
  vkCreateSampler(device_, &si, nullptr, &interop_sampler_);

  VkCommandBuffer init_cb = command_buffers_[0];
  VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(init_cb, &cbi);
  transition_image_layout(init_cb, interop_image_, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  vkEndCommandBuffer(init_cb);
  VkSubmitInfo s = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &init_cb};
  vkQueueSubmit(graphics_queue_, 1, &s, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue_);
}

void VulkanApp::create_descriptors() {
  VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1000};
  VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                   .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                   .maxSets = 1000,
                                   .poolSizeCount = 1,
                                   .pPoolSizes = &ps};
  vkCreateDescriptorPool(device_, &pi, nullptr, &descriptor_pool_);
  VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                    .descriptorPool = descriptor_pool_,
                                    .descriptorSetCount = 1,
                                    .pSetLayouts = &descriptor_set_layout_};
  vkAllocateDescriptorSets(device_, &ai, &descriptor_set_);
  VkDescriptorImageInfo di = {
      .sampler = interop_sampler_, .imageView = interop_view_, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set_,
                            .dstBinding = 0,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .pImageInfo = &di};
  vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
}

void VulkanApp::setup_cuda_interop() {
  int fd = export_memory_fd();
  size_t sz = current_width_ * current_height_ * 4 * sizeof(float);
  cuda_interop_pointer_ = import_vulkan_memory(fd, sz);
}

void VulkanApp::setup_imgui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsDark();

  ImGuiStyle& style = ImGui::GetStyle();
  for (int i = 0; i < ImGuiCol_COUNT; i++) {
    ImVec4& col = style.Colors[i];
    col.x = std::pow(col.x, 2.2f);
    col.y = std::pow(col.y, 2.2f);
    col.z = std::pow(col.z, 2.2f);
  }

  ImGui_ImplGlfw_InitForVulkan(window_, true);
  ImGui_ImplVulkan_InitInfo i = {};
  i.Instance = instance_;
  i.PhysicalDevice = physical_device_;
  i.Device = device_;
  i.QueueFamily = find_queue_families(physical_device_).graphics_family.value();
  i.Queue = graphics_queue_;
  i.DescriptorPool = descriptor_pool_;
  i.MinImageCount = kMaxFramesInFlight;
  i.ImageCount = kMaxFramesInFlight;
  i.PipelineInfoMain.RenderPass = render_pass_;
  ImGui_ImplVulkan_Init(&i);
  imgui_texture_ =
      ImGui_ImplVulkan_AddTexture(interop_sampler_, interop_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

int VulkanApp::export_memory_fd() {
  VkMemoryGetFdInfoKHR info = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                               .memory = interop_memory_,
                               .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  auto f = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
  int fd;
  f(device_, &info, &fd);
  return fd;
}

void VulkanApp::transition_image_layout(VkCommandBuffer cb, VkImage im, VkFormat f, VkImageLayout ol,
                                        VkImageLayout nl) {
  (void)f;
  VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            .srcAccessMask = 0,
                            .dstAccessMask = 0,
                            .oldLayout = ol,
                            .newLayout = nl,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = im,
                            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                 .baseMipLevel = 0,
                                                 .levelCount = 1,
                                                 .baseArrayLayer = 0,
                                                 .layerCount = 1}};
  VkPipelineStageFlags s = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, d = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  if (nl == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    s = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    d = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (ol == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    s = VK_PIPELINE_STAGE_TRANSFER_BIT;
    d = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (ol == VK_IMAGE_LAYOUT_UNDEFINED) {
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    s = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    d = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  vkCmdPipelineBarrier(cb, s, d, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void VulkanApp::cleanup() {
  should_stop_render_ = true;
  if (cpu_render_thread_.joinable()) cpu_render_thread_.join();
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  for (int i = 0; i < kMaxFramesInFlight; i++) {
    if (render_finished_semaphores_[i]) vkDestroySemaphore(device_, render_finished_semaphores_[i], nullptr);
    if (image_available_semaphores_[i]) vkDestroySemaphore(device_, image_available_semaphores_[i], nullptr);
    if (in_flight_fences_[i]) vkDestroyFence(device_, in_flight_fences_[i], nullptr);
  }
  if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
  for (auto f : framebuffers_) vkDestroyFramebuffer(device_, f, nullptr);
  if (graphics_pipeline_) vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
  if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
  if (render_pass_) vkDestroyRenderPass(device_, render_pass_, nullptr);
  for (auto v : swapchain_image_views_) vkDestroyImageView(device_, v, nullptr);
  if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
  if (interop_sampler_) vkDestroySampler(device_, interop_sampler_, nullptr);
  if (interop_view_) vkDestroyImageView(device_, interop_view_, nullptr);
  if (interop_image_) vkDestroyImage(device_, interop_image_, nullptr);
  if (interop_buffer_) vkDestroyBuffer(device_, interop_buffer_, nullptr);
  if (interop_memory_) vkFreeMemory(device_, interop_memory_, nullptr);
  if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
  if (descriptor_set_layout_) vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
  if (device_) vkDestroyDevice(device_, nullptr);
  if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
  if (instance_) vkDestroyInstance(instance_, nullptr);
  if (window_) glfwDestroyWindow(window_);
  cleanup_cuda_interop();
  glfwTerminate();
}

void VulkanApp::setup_world() {
  using std::make_shared;
  world_.clear();

  // Reset camera defaults
  camera_pos_[0] = 13.0f;
  camera_pos_[1] = 2.0f;
  camera_pos_[2] = 3.0f;
  camera_target_[0] = 0.0f;
  camera_target_[1] = 0.0f;
  camera_target_[2] = 0.0f;
  camera_fov_ = 20.0f;
  aspect_ratio_ = 16.0f / 9.0f;
  background_color_[0] = 0.70f;
  background_color_[1] = 0.80f;
  background_color_[2] = 1.00f;
  defocus_angle_ = 0.6f;
  focus_distance_ = 10.0f;

  if (scene_type_ == Scenes::STATIC) {
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world_.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));
    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        auto choose_mat = random_double();
        point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
        if ((center - point3(4, 0.2, 0)).length() > 0.9) {
          std::shared_ptr<material> sphere_material;
          if (choose_mat < 0.8) {
            auto albedo = color::random() * color::random();
            sphere_material = make_shared<lambertian>(albedo);
            world_.add(make_shared<sphere>(center, 0.2, sphere_material));
          } else if (choose_mat < 0.95) {
            auto albedo = color::random(0.5, 1);
            auto fuzz = random_double(0, 0.5);
            sphere_material = make_shared<metal>(albedo, fuzz);
            world_.add(make_shared<sphere>(center, 0.2, sphere_material));
          } else {
            sphere_material = make_shared<dielectric>(1.5);
            world_.add(make_shared<sphere>(center, 0.2, sphere_material));
          }
        }
      }
    }
    world_.add(make_shared<sphere>(point3(0, 1, 0), 1.0, make_shared<dielectric>(1.5)));
    world_.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, make_shared<lambertian>(color(0.4, 0.2, 0.1))));
    world_.add(make_shared<sphere>(point3(4, 1, 0), 1.0, make_shared<metal>(color(0.7, 0.6, 0.5), 0.0)));
  } else if (scene_type_ == Scenes::MOTION) {
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world_.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));
    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        auto choose_mat = random_double();
        point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
        if ((center - point3(4, 0.2, 0)).length() > 0.9) {
          std::shared_ptr<material> sphere_material;
          if (choose_mat < 0.8) {
            auto albedo = color::random() * color::random();
            sphere_material = make_shared<lambertian>(albedo);
            auto center2 = center + vec3(0, random_double(0, .5), 0);
            world_.add(make_shared<sphere>(center, center2, 0.2, sphere_material));
          } else if (choose_mat < 0.95) {
            auto albedo = color::random(0.5, 1);
            auto fuzz = random_double(0, 0.5);
            sphere_material = make_shared<metal>(albedo, fuzz);
            world_.add(make_shared<sphere>(center, 0.2, sphere_material));
          } else {
            sphere_material = make_shared<dielectric>(1.5);
            world_.add(make_shared<sphere>(center, 0.2, sphere_material));
          }
        }
      }
    }
    world_.add(make_shared<sphere>(point3(0, 1, 0), 1.0, make_shared<dielectric>(1.5)));
    world_.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, make_shared<lambertian>(color(0.4, 0.2, 0.1))));
    world_.add(make_shared<sphere>(point3(4, 1, 0), 1.0, make_shared<metal>(color(0.7, 0.6, 0.5), 0.0)));
  } else if (scene_type_ == Scenes::CHECKERED) {
    auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1), color(.9, .9, .9));
    world_.add(make_shared<sphere>(point3(0, -10, 0), 10, make_shared<lambertian>(checker)));
    world_.add(make_shared<sphere>(point3(0, 10, 0), 10, make_shared<lambertian>(checker)));
  } else if (scene_type_ == Scenes::EARTH) {
    auto earth_texture = make_shared<image_texture>("earthmap.jpg");
    world_.add(make_shared<sphere>(point3(0, 0, 0), 2, make_shared<lambertian>(earth_texture)));
  } else if (scene_type_ == Scenes::PERLIN) {
    auto pertext = make_shared<noise_texture>(5);
    world_.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(pertext)));
    world_.add(make_shared<sphere>(point3(0, 2, 0), 2, make_shared<lambertian>(pertext)));
  } else if (scene_type_ == Scenes::QUAD) {
    camera_pos_[0] = 0.0f;
    camera_pos_[1] = 0.0f;
    camera_pos_[2] = 9.0f;
    camera_fov_ = 80.0f;
    aspect_ratio_ = 1.0f;
    background_color_[0] = 0.70f;
    background_color_[1] = 0.80f;
    background_color_[2] = 1.00f;
    defocus_angle_ = 0.0f;
    world_.add(make_shared<quad>(point3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0),
                                 make_shared<lambertian>(color(1.0, 0.2, 0.2))));
    world_.add(make_shared<quad>(point3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0),
                                 make_shared<lambertian>(color(0.2, 1.0, 0.2))));
    world_.add(make_shared<quad>(point3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0),
                                 make_shared<lambertian>(color(0.2, 0.2, 1.0))));
    world_.add(make_shared<quad>(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4),
                                 make_shared<lambertian>(color(1.0, 0.5, 0.0))));
    world_.add(make_shared<quad>(point3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4),
                                 make_shared<lambertian>(color(0.2, 0.8, 0.8))));
  } else if (scene_type_ == Scenes::LIGHT) {
    camera_pos_[0] = 26.0f;
    camera_pos_[1] = 3.0f;
    camera_pos_[2] = 6.0f;
    camera_target_[0] = 0.0f;
    camera_target_[1] = 2.0f;
    camera_target_[2] = 0.0f;
    background_color_[0] = 0.0f;
    background_color_[1] = 0.0f;
    background_color_[2] = 0.0f;
    defocus_angle_ = 0.0f;
    auto pertext = make_shared<noise_texture>(4);
    world_.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(pertext)));
    world_.add(make_shared<sphere>(point3(0, 2, 0), 2, make_shared<lambertian>(pertext)));
    auto difflight = make_shared<diffuse_light>(color(4, 4, 4));
    world_.add(make_shared<sphere>(point3(0, 7, 0), 2, difflight));
    world_.add(make_shared<quad>(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), difflight));
  } else if (scene_type_ == Scenes::CORNELL) {
    camera_pos_[0] = 278.0f;
    camera_pos_[1] = 278.0f;
    camera_pos_[2] = -800.0f;
    camera_target_[0] = 278.0f;
    camera_target_[1] = 278.0f;
    camera_target_[2] = 0.0f;
    camera_fov_ = 40.0f;
    aspect_ratio_ = 1.0f;
    background_color_[0] = 0.0f;
    background_color_[1] = 0.0f;
    background_color_[2] = 0.0f;
    defocus_angle_ = 0.0f;
    auto red = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));
    world_.add(make_shared<quad>(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green));
    world_.add(make_shared<quad>(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red));
    world_.add(make_shared<quad>(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), light));
    world_.add(make_shared<quad>(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));
    world_.add(make_shared<quad>(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white));
    world_.add(make_shared<quad>(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white));
    std::shared_ptr<hittable> box1 = box(point3(0, 0, 0), point3(165, 330, 165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265, 0, 295));
    world_.add(box1);
    std::shared_ptr<hittable> box2 = box(point3(0, 0, 0), point3(165, 165, 165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130, 0, 65));
    world_.add(box2);
  } else if (scene_type_ == Scenes::SMOKE) {
    camera_pos_[0] = 278.0f;
    camera_pos_[1] = 278.0f;
    camera_pos_[2] = -800.0f;
    camera_target_[0] = 278.0f;
    camera_target_[1] = 278.0f;
    camera_target_[2] = 0.0f;
    camera_fov_ = 40.0f;
    aspect_ratio_ = 1.0f;
    background_color_[0] = 0.0f;
    background_color_[1] = 0.0f;
    background_color_[2] = 0.0f;
    defocus_angle_ = 0.0f;
    auto red = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(7, 7, 7));
    world_.add(make_shared<quad>(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green));
    world_.add(make_shared<quad>(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red));
    world_.add(make_shared<quad>(point3(113, 554, 127), vec3(330, 0, 0), vec3(0, 0, 305), light));
    world_.add(make_shared<quad>(point3(0, 555, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));
    world_.add(make_shared<quad>(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));
    world_.add(make_shared<quad>(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white));
    std::shared_ptr<hittable> b1 = box(point3(0, 0, 0), point3(165, 330, 165), white);
    b1 = make_shared<rotate_y>(b1, 15);
    b1 = make_shared<translate>(b1, vec3(265, 0, 295));
    std::shared_ptr<hittable> b2 = box(point3(0, 0, 0), point3(165, 165, 165), white);
    b2 = make_shared<rotate_y>(b2, -18);
    b2 = make_shared<translate>(b2, vec3(130, 0, 65));
    world_.add(make_shared<constant_medium>(b1, 0.01, color(0, 0, 0)));
    world_.add(make_shared<constant_medium>(b2, 0.01, color(1, 1, 1)));
  } else if (scene_type_ == Scenes::FINAL) {
    camera_pos_[0] = 478.0f;
    camera_pos_[1] = 278.0f;
    camera_pos_[2] = -600.0f;
    camera_target_[0] = 278.0f;
    camera_target_[1] = 278.0f;
    camera_target_[2] = 0.0f;
    camera_fov_ = 40.0f;
    aspect_ratio_ = 1.0f;
    background_color_[0] = 0.0f;
    background_color_[1] = 0.0f;
    background_color_[2] = 0.0f;
    defocus_angle_ = 0.0f;
    hittable_list boxes1;
    auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));
    for (int i = 0; i < 20; i++)
      for (int j = 0; j < 20; j++) {
        auto w = 100.0;
        auto x0 = -1000.0 + i * w, z0 = -1000.0 + j * w, y0 = 0.0;
        boxes1.add(box(point3(x0, y0, z0), point3(x0 + w, random_double(1, 101), z0 + w), ground));
      }
    world_.add(make_shared<bvh_node>(boxes1));
    auto light = make_shared<diffuse_light>(color(7, 7, 7));
    world_.add(make_shared<quad>(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light));
    auto center1 = point3(400, 400, 200), center2 = center1 + vec3(30, 0, 0);
    world_.add(make_shared<sphere>(center1, center2, 50, make_shared<lambertian>(color(0.7, 0.3, 0.1))));
    world_.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
    world_.add(make_shared<sphere>(point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)));
    auto boundary = make_shared<sphere>(point3(360, 150, 145), 70, make_shared<dielectric>(1.5));
    world_.add(boundary);
    world_.add(make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
    boundary = make_shared<sphere>(point3(0, 0, 0), 5000, make_shared<dielectric>(1.5));
    world_.add(make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));
    world_.add(make_shared<sphere>(point3(400, 200, 400), 100,
                                   make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"))));
    world_.add(
        make_shared<sphere>(point3(220, 280, 300), 80, make_shared<lambertian>(make_shared<noise_texture>(0.2))));
    hittable_list boxes2;
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    for (int j = 0; j < 1000; j++) boxes2.add(make_shared<sphere>(point3::random(0, 165), 10, white));
    world_.add(make_shared<translate>(make_shared<rotate_y>(make_shared<bvh_node>(boxes2), 15), vec3(-100, 270, 395)));
  } else if (scene_type_ == Scenes::CUSTOM) {
    camera_pos_[0] = 13.0f;
    camera_pos_[1] = 2.0f;
    camera_pos_[2] = 3.0f;
    camera_target_[0] = 0.0f;
    camera_target_[1] = 0.0f;
    camera_target_[2] = 0.0f;
    camera_fov_ = 20.0f;
    background_color_[0] = 0.0f;
    background_color_[1] = 0.0f;
    background_color_[2] = 0.0f;
    aspect_ratio_ = 1.0f;
    defocus_angle_ = 0.0f;
    auto checker = make_shared<checker_texture>(0.32, color(.2, .3, .1), color(.9, .9, .9));
    world_.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(checker)));
    auto pertext = make_shared<noise_texture>(1.5);
    std::shared_ptr<hittable> central_pillar =
        box(point3(-1, 0, -1), point3(1, 1, 1), make_shared<lambertian>(pertext));
    world_.add(central_pillar);
    std::shared_ptr<hittable> prism_boundary =
        box(point3(-1, 1.1, -1), point3(1, 2.5, 1), make_shared<dielectric>(1.5));
    world_.add(make_shared<constant_medium>(prism_boundary, 0.1, color(0.8, 0.8, 1.0)));
    auto earth_texture = make_shared<image_texture>("earthmap.jpg");
    world_.add(make_shared<sphere>(point3(0, 1.8, 0), 0.4, make_shared<lambertian>(earth_texture)));
    auto light_mat = make_shared<diffuse_light>(color(10, 10, 10));
    world_.add(make_shared<sphere>(point3(0, 1.8, 0), 0.1, light_mat));
    auto metal_mat = make_shared<metal>(color(0.8, 0.8, 0.8), 0.0);
    for (int i = 0; i < 4; i++) {
      float angle = i * 90.0f, rad = angle * 3.14159f / 180.0f;
      float x = 4.0f * cosf(rad), z = 4.0f * sinf(rad);
      std::shared_ptr<hittable> cp = box(point3(-0.5, 0, -0.5), point3(0.5, 3.0, 0.5), metal_mat);
      cp = make_shared<rotate_y>(cp, angle + 45);
      cp = make_shared<translate>(cp, vec3(x, 0, z));
      world_.add(cp);
    }
    auto blur_mat = make_shared<lambertian>(color(0.7, 0.3, 0.1));
    for (int i = 0; i < 5; i++) {
      point3 c1(-5, 1 + i, 5 - i * 2), c2(-3, 1 + i, 5 - i * 2);
      world_.add(make_shared<sphere>(c1, c2, 0.2, blur_mat));
    }
    world_.add(make_shared<quad>(point3(-5, 10, -5), vec3(10, 0, 0), vec3(0, 0, 10),
                                 make_shared<diffuse_light>(color(4, 4, 4))));
  }

  gpu_bvh_nodes_.clear();
  gpu_primitives_.clear();
  gpu_materials_.clear();
  gpu_textures_.clear();
  gpu_perlin_.clear();
  gpu_image_buffer_.clear();
  std::unordered_map<material*, int> mm;
  std::unordered_map<texture*, int> tm;
  flatten_hittable(std::make_shared<bvh_node>(world_), gpu_bvh_nodes_, gpu_primitives_, gpu_materials_, gpu_textures_,
                   gpu_perlin_, gpu_image_buffer_, mm, tm);
}

void VulkanApp::setup_camera() {
  cam_.aspect_ratio = aspect_ratio_;
  cam_.image_width = current_width_;
  cam_.samples_per_pixel = samples_per_pixel_;
  cam_.max_depth = max_depth_;
  cam_.lookfrom = point3(camera_pos_[0], camera_pos_[1], camera_pos_[2]);
  cam_.lookat = point3(camera_target_[0], camera_target_[1], camera_target_[2]);
  cam_.vup = vec3(0, 1, 0);
  cam_.vfov = camera_fov_;
  cam_.defocus_angle = defocus_angle_;
  cam_.focus_dist = focus_distance_;
  cam_.background = color(background_color_[0], background_color_[1], background_color_[2]);
}

QueueFamilyIndices VulkanApp::find_queue_families(VkPhysicalDevice d) {
  QueueFamilyIndices indices;
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(d, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(d, &count, families.data());
  for (int i = 0; i < (int)count; i++) {
    if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphics_family = i;
    VkBool32 present = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface_, &present);
    if (present) indices.present_family = i;
    if (indices.is_complete()) break;
  }
  return indices;
}

bool VulkanApp::is_device_suitable(VkPhysicalDevice d) { return find_queue_families(d).is_complete(); }
bool VulkanApp::check_device_extension_support(VkPhysicalDevice) { return true; }
SwapchainSupportDetails VulkanApp::query_swapchain_support(VkPhysicalDevice d) {
  SwapchainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d, surface_, &details.capabilities);
  uint32_t count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(d, surface_, &count, nullptr);
  if (count != 0) {
    details.formats.resize(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(d, surface_, &count, details.formats.data());
  }
  vkGetPhysicalDeviceSurfacePresentModesKHR(d, surface_, &count, nullptr);
  if (count != 0) {
    details.present_modes.resize(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(d, surface_, &count, details.present_modes.data());
  }
  return details;
}

VkSurfaceFormatKHR VulkanApp::choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& f) {
  for (const auto& fmt : f)
    if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return fmt;
  return f[0];
}
VkPresentModeKHR VulkanApp::choose_swap_present_mode(const std::vector<VkPresentModeKHR>&) {
  return VK_PRESENT_MODE_FIFO_KHR;
}
VkExtent2D VulkanApp::choose_swap_extent(const VkSurfaceCapabilitiesKHR& c) {
  if (c.currentExtent.width != std::numeric_limits<uint32_t>::max()) return c.currentExtent;
  int w, h;
  glfwGetFramebufferSize(window_, &w, &h);
  VkExtent2D actual = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
  actual.width = std::clamp(actual.width, c.minImageExtent.width, c.maxImageExtent.width);
  actual.height = std::clamp(actual.height, c.minImageExtent.height, c.maxImageExtent.height);
  return actual;
}
std::vector<const char*> VulkanApp::get_required_extensions() {
  uint32_t c = 0;
  const char** e = glfwGetRequiredInstanceExtensions(&c);
  std::vector<const char*> r(e, e + c);
  return r;
}
std::vector<char> VulkanApp::read_spirv(const std::string& f) {
  std::ifstream fi(f, std::ios::ate | std::ios::binary);
  size_t s = (size_t)fi.tellg();
  std::vector<char> b(s);
  fi.seekg(0);
  fi.read(b.data(), s);
  return b;
}
VkShaderModule VulkanApp::create_shader_module(const std::vector<char>& c) {
  VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                 .codeSize = c.size(),
                                 .pCode = reinterpret_cast<const uint32_t*>(c.data())};
  VkShaderModule m;
  vkCreateShaderModule(device_, &ci, nullptr, &m);
  return m;
}
VKAPI_ATTR VkBool32 VKAPI_CALL VulkanApp::debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                                         VkDebugUtilsMessageTypeFlagsEXT,
                                                         const VkDebugUtilsMessengerCallbackDataEXT*, void*) {
  return VK_FALSE;
}
void VulkanApp::setup_debug_messenger() {}
void VulkanApp::export_ppm() {
  int width = current_width_;
  int height = current_height_;
  std::vector<float4> host_buffer(width * height);

  // If we are in headless mode, cuda_interop_pointer_ was allocated via cudaMalloc
  // If we are in GUI mode, it was mapped from Vulkan
  cudaMemcpy(host_buffer.data(), cuda_interop_pointer_, width * height * sizeof(float4), cudaMemcpyDeviceToHost);

  std::ofstream ofs("output.ppm");
  ofs << "P3\n" << width << " " << height << "\n255\n";
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      float4 pixel = host_buffer[j * width + i];
      int r = int(255.999 * std::clamp(pixel.x, 0.0f, 1.0f));
      int g = int(255.999 * std::clamp(pixel.y, 0.0f, 1.0f));
      int b = int(255.999 * std::clamp(pixel.z, 0.0f, 1.0f));
      ofs << r << " " << g << " " << b << "\n";
    }
  }
  std::cout << "Render saved to output.ppm" << std::endl;
}

void VulkanApp::run_headless() {
  std::cout << "Starting headless render (" << current_width_ << "x" << current_height_ << ")..." << std::endl;

  // Manual CUDA allocation since we skip Vulkan interop in headless mode
  size_t buffer_size = current_width_ * current_height_ * sizeof(float4);
  cudaMalloc(&cuda_interop_pointer_, buffer_size);

  RenderConfig config;
  config.frame_buffer = (vec3_gpu*)cuda_interop_pointer_;
  config.width = current_width_;
  config.height = current_height_;
  config.samples_per_pixel = samples_per_pixel_;
  config.max_depth = max_depth_;
  config.background = Vec3f{background_color_[0], background_color_[1], background_color_[2]};
  config.lookfrom = Vec3f{camera_pos_[0], camera_pos_[1], camera_pos_[2]};
  config.lookat = Vec3f{camera_target_[0], camera_target_[1], camera_target_[2]};
  config.vup = Vec3f{0, 1, 0};
  config.vfov = camera_fov_;
  config.defocus_angle = defocus_angle_;
  config.focus_dist = focus_distance_;

  span<LinearBVHNode> bvh = {gpu_bvh_nodes_.data(), gpu_bvh_nodes_.size()};
  span<PrimitiveGPU> p_buf = {gpu_primitives_.data(), gpu_primitives_.size()};
  span<MaterialGPU> m_buf = {gpu_materials_.data(), gpu_materials_.size()};
  span<TextureGPU> t_buf = {gpu_textures_.data(), gpu_textures_.size()};
  span<PerlinDataGPU> per_buf = {gpu_perlin_.data(), gpu_perlin_.size()};
  span<unsigned char> i_buf = {gpu_image_buffer_.data(), gpu_image_buffer_.size()};

  auto start = std::chrono::high_resolution_clock::now();

  launch_render(config, bvh, p_buf, m_buf, t_buf, per_buf, i_buf);
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  float duration = std::chrono::duration<float>(end - start).count();
  std::cout << "Render completed in " << duration << "s" << std::endl;

  export_ppm();

  cudaFree(cuda_interop_pointer_);
}
bool VulkanApp::check_validation_layer_support() { return true; }
