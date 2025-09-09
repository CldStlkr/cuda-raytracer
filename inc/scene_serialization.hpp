#pragma once
#include "cuda_structs.hpp"
#include "gpu_type_conversion.hpp"
#include "hittable_list.hpp"
#include "material.hpp"
#include "sphere.hpp"
#include <memory>
#include <vector>

inline void serialize_scene(const hittable_list& world,
                            std::vector<SphereGPU>& spheres_out,
                            std::vector<MaterialGPU>& materials_out) {
  int material_id = 0;
  for (const auto& obj : world.objects) {
    if (auto s = std::dynamic_pointer_cast<sphere>(obj)) {
      auto m = s->get_material();

      if (auto lam = std::dynamic_pointer_cast<lambertian>(m)) {
        materials_out.push_back(to_gpu_material(*lam));
      } else if (auto met = std::dynamic_pointer_cast<metal>(m)) {
        materials_out.push_back(to_gpu_material(*met));
      } else if (auto die = std::dynamic_pointer_cast<dielectric>(m)) {
        materials_out.push_back(to_gpu_material(*die));
      }

      spheres_out.push_back(to_gpu_sphere(*s, material_id));
      material_id++;
    }
  }
}
