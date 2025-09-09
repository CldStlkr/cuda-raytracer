#include "cuda_structs.hpp"
#include "material.hpp"
#include "sphere.hpp"

inline SphereGPU to_gpu_sphere(const sphere& s, int material_id) {
  auto c = s.get_center();
  return SphereGPU{Vec3f{static_cast<float>(c.x()), static_cast<float>(c.y()),
                         static_cast<float>(c.z())},
                   static_cast<float>(s.get_radius()), material_id

  };
}

// flatten vec3 to Vec3f
inline Vec3f to_gpu_vec3(const vec3& v) {
  return Vec3f{
      static_cast<float>(v.x()),
      static_cast<float>(v.y()),
      static_cast<float>(v.z()),
  };
}

inline MaterialGPU to_gpu_material(const lambertian& m) {
  return MaterialGPU{LAMBERTIAN, to_gpu_vec3(m.get_albedo()), 0.0f, 0.0f};
}

inline MaterialGPU to_gpu_material(const metal& m) {
  return MaterialGPU{METAL, to_gpu_vec3(m.get_albedo()),
                     static_cast<float>(m.get_fuzz()), 0.0f};
}

inline MaterialGPU to_gpu_material(const dielectric& m) {
  return MaterialGPU{DIELECTRIC,
                     {1.0f, 1.0f, 1.0f},
                     0.0f,
                     static_cast<float>(m.get_refraction_index())};
}
