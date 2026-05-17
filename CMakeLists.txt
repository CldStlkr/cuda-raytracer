cmake_minimum_required(VERSION 3.20)
project(ray_gui LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
enable_language(C)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------- Vulkan ----------
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/vulkansdk/1.4.341.1/x86_64")
find_package(Vulkan REQUIRED)
if(NOT Vulkan_GLSLC_EXECUTABLE)
    set(
      Vulkan_GLSLC_EXECUTABLE
      "$ENV{HOME}/vulkansdk/1.4.341.1/x86_64/bin/glslc"
    )
endif()
message(STATUS "glslc: ${Vulkan_GLSLC_EXECUTABLE}")

# ---------- GLFW & CUDA ----------
find_package(glfw3 3.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ---------- ImGui (Vulkan backend) ----------
file(GLOB IMGUI_SRC
    extern/imgui/*.cpp
    extern/imgui/backends/imgui_impl_glfw.cpp
    extern/imgui/backends/imgui_impl_vulkan.cpp
)
add_library(imgui STATIC ${IMGUI_SRC})
target_include_directories(imgui PUBLIC extern/imgui extern/imgui/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

# ---------- SPIR-V shader compilation ----------
set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BIN_DIR})
add_custom_command(
    OUTPUT  ${SHADER_BIN_DIR}/fullscreen.vert.spv
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${SHADER_SRC_DIR}/fullscreen.vert
              -o ${SHADER_BIN_DIR}/fullscreen.vert.spv
    DEPENDS ${SHADER_SRC_DIR}/fullscreen.vert
    COMMENT "Compiling fullscreen.vert → SPIR-V"
)
add_custom_command(
    OUTPUT  ${SHADER_BIN_DIR}/tonemap.frag.spv
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${SHADER_SRC_DIR}/tonemap.frag
              -o ${SHADER_BIN_DIR}/tonemap.frag.spv
    DEPENDS ${SHADER_SRC_DIR}/tonemap.frag
    COMMENT "Compiling tonemap.frag → SPIR-V"
)
add_custom_target(shaders ALL DEPENDS
    ${SHADER_BIN_DIR}/fullscreen.vert.spv
    ${SHADER_BIN_DIR}/tonemap.frag.spv
)

# ---------- Executable ----------
add_executable(main
    src/main.cpp
    src/vulkan_app.cpp
    src/cuda_renderer.cu
    src/cuda_structs.cpp
    src/stb_image_impl.cpp
)
add_dependencies(main shaders)

if(EXISTS ${CMAKE_SOURCE_DIR}/extern/cccl)
    target_include_directories(main BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/cccl/libcudacxx/include
    )
endif()
target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/extern/stb-image
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(main PRIVATE
    SHADER_DIR="${SHADER_BIN_DIR}/"
)
target_link_libraries(main PRIVATE
    imgui
    glfw
    Vulkan::Vulkan
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::curand
)
if(MSVC)
    target_compile_options(main PRIVATE /W4)
else()
    target_compile_options(
        main PRIVATE -Wall -Wextra -Wno-missing-field-initializers
    )
endif()
