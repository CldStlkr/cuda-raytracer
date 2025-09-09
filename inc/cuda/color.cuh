#ifndef COLOR_H
#define COLOR_H

#include "vec.cuh"
#include <cuda_runtime.h>
#include <iostream>

using color = vec3_gpu;

// Writes normalized [0,1] color as [0,255] integers
__host__ inline void write_color(std::ostream& out, const color& pixel_color) {
  float r = pixel_color.x();
  float g = pixel_color.y();
  float b = pixel_color.z();

  // Clamp just in case of overshoot
  r = fminf(fmaxf(r, 0.0f), 0.999f);
  g = fminf(fmaxf(g, 0.0f), 0.999f);
  b = fminf(fmaxf(b, 0.0f), 0.999f);

  int rbyte = static_cast<int>(255.999f * r);
  int gbyte = static_cast<int>(255.999f * g);
  int bbyte = static_cast<int>(255.999f * b);

  out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif
