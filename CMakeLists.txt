cmake_minimum_required(VERSION 3.20)

project(ray_gui LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

enable_language(C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)


find_package(CUDAToolkit REQUIRED)

# ---------- GLAD ----------
add_library(glad STATIC extern/glad/src/gl.c)
target_include_directories(glad PUBLIC extern/glad/include)

# ---------- ImGui ----------
file(GLOB IMGUI_SRC
    extern/imgui/*.cpp
    extern/imgui/backends/imgui_impl_glfw.cpp
    extern/imgui/backends/imgui_impl_opengl3.cpp
)
add_library(imgui STATIC ${IMGUI_SRC})
target_include_directories(imgui PUBLIC extern/imgui extern/imgui/backends)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)

# ---------- Executable ----------
add_executable(main
    src/main.cpp
    src/cuda_renderer.cu
)

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/inc)

# Link libraries with proper order
target_link_libraries(main PRIVATE
    imgui
    glad
    glfw
    OpenGL::GL
    CUDA::cudart
)

# Compiler-specific options
if(MSVC)
  target_compile_options(main PRIVATE /W4)
else()
  target_compile_options(main PRIVATE -Wall -Wextra)
endif()
