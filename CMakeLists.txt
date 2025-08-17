cmake_minimum_required(VERSION 3.20)
project(ray_gui LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable C language for GLAD
enable_language(C)

# Find packages first
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

# Include directories
include_directories(inc)

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
add_executable(main main.cpp)

# Link libraries with proper order
target_link_libraries(main PRIVATE
    imgui
    glad
    glfw
    OpenGL::GL
)

# Compiler-specific options
if(MSVC)
  target_compile_options(main PRIVATE /W4)
else()
  target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic)
endif()
