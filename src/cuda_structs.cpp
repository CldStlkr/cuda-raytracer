#include "cuda_structs.hpp"
#include "constant_medium.hpp"
#include "material.hpp"
#include "texture.hpp"

// Define CumTransform locally
struct CumTransform {
  vec3 offset{0, 0, 0};
  double sin_t{0};
  double cos_t{1};
  bool has_rot{false};

  void apply_translate(const vec3 &t) { offset += t; }

  void apply_rotate_y(double angle_deg) {
    auto radians = degrees_to_radians(angle_deg);
    double s = std::sin(radians);
    double c = std::cos(radians);
    // Combine rotations (angles just add around Y axis)
    if (!has_rot) {
      sin_t = s;
      cos_t = c;
      has_rot = true;
    } else {
      double new_s = sin_t * c + cos_t * s;
      double new_c = cos_t * c - sin_t * s;
      sin_t = new_s;
      cos_t = new_c;
    }
  }

  point3 apply(const point3 &p) const {
    point3 pr = p;
    if (has_rot) {
      pr = point3(cos_t * p.x() + sin_t * p.z(), p.y(),
                  -sin_t * p.x() + cos_t * p.z());
    }
    return pr + offset;
  }

  vec3 apply_vec(const vec3 &v) const {
    if (!has_rot)
      return v;
    return vec3(cos_t * v.x() + sin_t * v.z(), v.y(),
                -sin_t * v.x() + cos_t * v.z());
  }

  aabb transform_bbox(const aabb &box) const {
    if (box.x.min >= infinity)
      return box;
    point3 min(infinity, infinity, infinity);
    point3 max(-infinity, -infinity, -infinity);
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          double x = i * box.x.max + (1 - i) * box.x.min;
          double y = j * box.y.max + (1 - j) * box.y.min;
          double z = k * box.z.max + (1 - k) * box.z.min;
          point3 tester = apply(point3(x, y, z));
          for (int c = 0; c < 3; c++) {
            min[c] = std::fmin(min[c], tester[c]);
            max[c] = std::fmax(max[c], tester[c]);
          }
        }
      }
    }
    return aabb(min, max);
  }
};

int get_or_add_texture(std::shared_ptr<texture> tex_ptr,
                       std::vector<TextureGPU> &linear_textures,
                       std::vector<PerlinDataGPU> &linear_perlin,
                       std::vector<unsigned char> &image_buffer,
                       std::unordered_map<texture *, int> &tex_map) {
  if (!tex_ptr)
    return -1;
  auto it = tex_map.find(tex_ptr.get());
  if (it != tex_map.end())
    return it->second;

  TextureGPU gpu_tex;

  if (auto solid = dynamic_cast<solid_color *>(tex_ptr.get())) {
    gpu_tex.type = TextureType::SOLID;
    gpu_tex.solid.color = to_vec3f(solid->albedo);
  } else if (auto checker = dynamic_cast<checker_texture *>(tex_ptr.get())) {
    gpu_tex.type = TextureType::CHECKER;
    gpu_tex.checker.inv_scale = static_cast<float>(checker->inv_scale);
    gpu_tex.checker.even_tex_idx = get_or_add_texture(
        checker->even, linear_textures, linear_perlin, image_buffer, tex_map);
    gpu_tex.checker.odd_tex_idx = get_or_add_texture(
        checker->odd, linear_textures, linear_perlin, image_buffer, tex_map);
  } else if (auto img = dynamic_cast<image_texture *>(tex_ptr.get())) {
    gpu_tex.type = TextureType::IMAGE;
    int w = img->image.width();
    int h = img->image.height();
    gpu_tex.image.width = w;
    gpu_tex.image.height = h;
    gpu_tex.image.bytes_per_scanline = w * 3;
    gpu_tex.image.offset = image_buffer.size();
    if (w > 0 && h > 0) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const unsigned char *pixel = img->image.pixel_data(x, y);
          image_buffer.push_back(pixel[0]);
          image_buffer.push_back(pixel[1]);
          image_buffer.push_back(pixel[2]);
        }
      }
    }
  } else if (auto noise_tex = dynamic_cast<noise_texture *>(tex_ptr.get())) {
    gpu_tex.type = TextureType::NOISE;
    gpu_tex.noise.scale = static_cast<float>(noise_tex->scale);
    gpu_tex.noise.perlin_data_idx = linear_perlin.size();

    PerlinDataGPU p_data;
    for (int i = 0; i < 256; ++i) {
      p_data.randvec[i] = to_vec3f(noise_tex->noise.randvec[i]);
      p_data.perm_x[i] = noise_tex->noise.perm_x[i];
      p_data.perm_y[i] = noise_tex->noise.perm_y[i];
      p_data.perm_z[i] = noise_tex->noise.perm_z[i];
    }
    linear_perlin.push_back(p_data);
  }

  int new_id = linear_textures.size();
  linear_textures.push_back(gpu_tex);
  tex_map[tex_ptr.get()] = new_id;
  return new_id;
}

int get_or_add_material(std::shared_ptr<material> mat_ptr,
                        std::vector<MaterialGPU> &linear_materials,
                        std::vector<TextureGPU> &linear_textures,
                        std::vector<PerlinDataGPU> &linear_perlin,
                        std::vector<unsigned char> &image_buffer,
                        std::unordered_map<material *, int> &mat_map,
                        std::unordered_map<texture *, int> &tex_map) {
  if (!mat_ptr)
    return -1;
  auto it = mat_map.find(mat_ptr.get());
  if (it != mat_map.end())
    return it->second;

  MaterialGPU gpu_mat;
  if (auto lambert = dynamic_cast<lambertian *>(mat_ptr.get())) {
    gpu_mat.type = MaterialType::LAMBERTIAN;
    gpu_mat.albedo_tex_id = get_or_add_texture(
        lambert->tex, linear_textures, linear_perlin, image_buffer, tex_map);
  } else if (auto met = dynamic_cast<metal *>(mat_ptr.get())) {
    gpu_mat.type = MaterialType::METAL;
    TextureGPU gpu_tex;
    gpu_tex.type = TextureType::SOLID;
    gpu_tex.solid.color = to_vec3f(met->get_albedo());
    gpu_mat.albedo_tex_id = linear_textures.size();
    linear_textures.push_back(gpu_tex);
    gpu_mat.fuzz = static_cast<float>(met->get_fuzz());
  } else if (auto die = dynamic_cast<dielectric *>(mat_ptr.get())) {
    gpu_mat.type = MaterialType::DIELECTRIC;
    TextureGPU gpu_tex;
    gpu_tex.type = TextureType::SOLID;
    gpu_tex.solid.color = Vec3f{1.0f, 1.0f, 1.0f};
    gpu_mat.albedo_tex_id = linear_textures.size();
    linear_textures.push_back(gpu_tex);
    gpu_mat.ref_idx = static_cast<float>(die->get_refraction_index());
  } else if (auto diffuse = dynamic_cast<diffuse_light *>(mat_ptr.get())) {
    gpu_mat.type = MaterialType::DIFFUSE_LIGHT;
    gpu_mat.albedo_tex_id = get_or_add_texture(
        diffuse->tex, linear_textures, linear_perlin, image_buffer, tex_map);
  } else if (auto iso = dynamic_cast<isotropic *>(mat_ptr.get())) {
    gpu_mat.type = MaterialType::LAMBERTIAN; // Map isotropic to lambertian for
                                             // volume scattering logic handling
    gpu_mat.albedo_tex_id = get_or_add_texture(
        iso->tex, linear_textures, linear_perlin, image_buffer, tex_map);
  }

  int new_id = linear_materials.size();
  linear_materials.push_back(gpu_mat);
  mat_map[mat_ptr.get()] = new_id;
  return new_id;
}

// Forward declaration
int flatten_hittable_internal(std::shared_ptr<hittable> node,
                              std::vector<LinearBVHNode> &linear_nodes,
                              std::vector<PrimitiveGPU> &linear_primitives,
                              std::vector<MaterialGPU> &linear_materials,
                              std::vector<TextureGPU> &linear_textures,
                              std::vector<PerlinDataGPU> &linear_perlin,
                              std::vector<unsigned char> &image_buffer,
                              std::unordered_map<material *, int> &mat_map,
                              std::unordered_map<texture *, int> &tex_map,
                              CumTransform current_trans);

int flatten_hittable(std::shared_ptr<hittable> node,
                     std::vector<LinearBVHNode> &linear_nodes,
                     std::vector<PrimitiveGPU> &linear_primitives,
                     std::vector<MaterialGPU> &linear_materials,
                     std::vector<TextureGPU> &linear_textures,
                     std::vector<PerlinDataGPU> &linear_perlin,
                     std::vector<unsigned char> &image_buffer,
                     std::unordered_map<material *, int> &mat_map,
                     std::unordered_map<texture *, int> &tex_map) {

  return flatten_hittable_internal(
      node, linear_nodes, linear_primitives, linear_materials, linear_textures,
      linear_perlin, image_buffer, mat_map, tex_map, CumTransform());
}

int flatten_hittable_internal(std::shared_ptr<hittable> node,
                              std::vector<LinearBVHNode> &linear_nodes,
                              std::vector<PrimitiveGPU> &linear_primitives,
                              std::vector<MaterialGPU> &linear_materials,
                              std::vector<TextureGPU> &linear_textures,
                              std::vector<PerlinDataGPU> &linear_perlin,
                              std::vector<unsigned char> &image_buffer,
                              std::unordered_map<material *, int> &mat_map,
                              std::unordered_map<texture *, int> &tex_map,
                              CumTransform current_trans) {

  // Process Pass-through Wrappers
  if (auto t_node = dynamic_cast<translate *>(node.get())) {
    current_trans.apply_translate(t_node->offset);
    return flatten_hittable_internal(
        t_node->object, linear_nodes, linear_primitives, linear_materials,
        linear_textures, linear_perlin, image_buffer, mat_map, tex_map,
        current_trans);
  }
  if (auto r_node = dynamic_cast<rotate_y *>(node.get())) {
    // Note, rotate_y stores sin_theta and cos_theta. We can just derive angle
    // from it or pass sin/cos directly. wait, our CumTransform holds sin and
    // cos directly. Let's just modify the struct or use math.atan2? Let's use
    // atan2 for safety.
    double angle = std::atan2(r_node->sin_theta, r_node->cos_theta) * 180.0 /
                   3.1415926535897932385;
    current_trans.apply_rotate_y(angle);
    return flatten_hittable_internal(
        r_node->object, linear_nodes, linear_primitives, linear_materials,
        linear_textures, linear_perlin, image_buffer, mat_map, tex_map,
        current_trans);
  }
  if (auto hl = dynamic_cast<hittable_list *>(node.get())) {
    auto bvh_wrap = std::make_shared<bvh_node>(*hl);
    return flatten_hittable_internal(bvh_wrap, linear_nodes, linear_primitives,
                                     linear_materials, linear_textures,
                                     linear_perlin, image_buffer, mat_map,
                                     tex_map, current_trans);
  }

  // Handle actual primitives & BVH
  if (auto bvh = dynamic_cast<bvh_node *>(node.get())) {
    if (bvh->left() == bvh->right()) {
      return flatten_hittable_internal(
          bvh->left(), linear_nodes, linear_primitives, linear_materials,
          linear_textures, linear_perlin, image_buffer, mat_map, tex_map,
          current_trans);
    }

    int curr_idx = linear_nodes.size();
    linear_nodes.emplace_back();

    flatten_hittable_internal(bvh->left(), linear_nodes, linear_primitives,
                              linear_materials, linear_textures, linear_perlin,
                              image_buffer, mat_map, tex_map, current_trans);

    int right_offset = flatten_hittable_internal(
        bvh->right(), linear_nodes, linear_primitives, linear_materials,
        linear_textures, linear_perlin, image_buffer, mat_map, tex_map,
        current_trans);

    aabb box = current_trans.transform_bbox(bvh->bounding_box());
    double delta = 0.0001;
    if (box.x.max - box.x.min < delta) {
      box.x.min -= delta / 2.0;
      box.x.max += delta / 2.0;
    }
    if (box.y.max - box.y.min < delta) {
      box.y.min -= delta / 2.0;
      box.y.max += delta / 2.0;
    }
    if (box.z.max - box.z.min < delta) {
      box.z.min -= delta / 2.0;
      box.z.max += delta / 2.0;
    }
    linear_nodes[curr_idx].aabb_min =
        Vec3f{float(box.x.min), float(box.y.min), float(box.z.min)};
    linear_nodes[curr_idx].aabb_max =
        Vec3f{float(box.x.max), float(box.y.max), float(box.z.max)};
    linear_nodes[curr_idx].n_primitives = 0;
    linear_nodes[curr_idx].second_child_offset = right_offset;

    return curr_idx;

  } else {
    int curr_idx = linear_nodes.size();
    linear_nodes.emplace_back();

    int prim_idx = linear_primitives.size();
    PrimitiveGPU prim;

    if (auto c_med = dynamic_cast<constant_medium *>(node.get())) {
      // It's a volume! Extract boundary.
      // We look at the immediate boundary to see if it's a sphere or
      // translated/rotated hittable_list (box)
      std::shared_ptr<hittable> unwrap = c_med->boundary;
      CumTransform vol_trans = current_trans;

      while (auto t = dynamic_cast<translate *>(unwrap.get())) {
        vol_trans.apply_translate(t->offset);
        unwrap = t->object;
      }
      // Note rotate_y wrapper could be inside or outside.
      // Actually simply using dynamic_cast iteratively.
      bool drilling = true;
      while (drilling) {
        if (auto t = dynamic_cast<translate *>(unwrap.get())) {
          vol_trans.apply_translate(t->offset);
          unwrap = t->object;
        } else if (auto r = dynamic_cast<rotate_y *>(unwrap.get())) {
          double angle = std::atan2(r->sin_theta, r->cos_theta) * 180.0 /
                         3.1415926535897932385;
          vol_trans.apply_rotate_y(angle);
          unwrap = r->object;
        } else {
          drilling = false;
        }
      }

      if (auto sph = dynamic_cast<sphere *>(unwrap.get())) {
        prim.type = PrimitiveType::VOLUME_SPHERE;
        prim.volume_sphere.center =
            to_vec3f(vol_trans.apply(sph->get_center().origin()));
        prim.volume_sphere.radius = static_cast<float>(sph->get_radius());
        prim.volume_sphere.neg_inv_density =
            static_cast<float>(c_med->neg_inv_density);
      } else if (auto hl = dynamic_cast<hittable_list *>(unwrap.get())) {
        // It's a box! Find local AABB.
        prim.type = PrimitiveType::VOLUME_BOX;
        aabb local_box = hl->bounding_box();
        prim.volume_box.local_min =
            to_vec3f(vec3(local_box.x.min, local_box.y.min, local_box.z.min));
        prim.volume_box.local_max =
            to_vec3f(vec3(local_box.x.max, local_box.y.max, local_box.z.max));
        prim.volume_box.offset = to_vec3f(vol_trans.offset);
        prim.volume_box.sin_theta = static_cast<float>(vol_trans.sin_t);
        prim.volume_box.cos_theta = static_cast<float>(vol_trans.cos_t);
        prim.volume_box.neg_inv_density =
            static_cast<float>(c_med->neg_inv_density);
      }

      prim.material_id = get_or_add_material(
          c_med->phase_function, linear_materials, linear_textures,
          linear_perlin, image_buffer, mat_map, tex_map);

    } else if (auto sphere_ptr = dynamic_cast<sphere *>(node.get())) {
      vec3 dist_vec = sphere_ptr->get_center().direction();
      if (dist_vec.length_squared() > 1e-8) {
        prim.type = PrimitiveType::MOVING_SPHERE;
        // The start and vec map over time in World Space natively!
        prim.moving_sphere.center_start =
            to_vec3f(current_trans.apply(sphere_ptr->get_center().origin()));
        prim.moving_sphere.center_vec =
            to_vec3f(current_trans.apply_vec(dist_vec));
        prim.moving_sphere.radius =
            static_cast<float>(sphere_ptr->get_radius());
      } else {
        prim.type = PrimitiveType::SPHERE;
        prim.sphere.center =
            to_vec3f(current_trans.apply(sphere_ptr->get_center().origin()));
        prim.sphere.radius = static_cast<float>(sphere_ptr->get_radius());
      }

      prim.material_id = get_or_add_material(
          sphere_ptr->get_material(), linear_materials, linear_textures,
          linear_perlin, image_buffer, mat_map, tex_map);

    } else if (auto quad_ptr = dynamic_cast<quad *>(node.get())) {
      prim.type = PrimitiveType::QUAD;
      prim.quad.Q = to_vec3f(current_trans.apply(quad_ptr->Q));
      prim.quad.u = to_vec3f(current_trans.apply_vec(quad_ptr->u));
      prim.quad.v = to_vec3f(current_trans.apply_vec(quad_ptr->v));

      [[maybe_unused]] vec3 w = quad_ptr->w;
      [[maybe_unused]] vec3 n = quad_ptr->normal;

      // recalculate w and normal properly in world space!
      vec3 u_w = current_trans.apply_vec(quad_ptr->u);
      vec3 v_w = current_trans.apply_vec(quad_ptr->v);
      vec3 n_w = cross(u_w, v_w);
      vec3 normal_w = unit_vector(n_w);
      point3 Q_w = current_trans.apply(quad_ptr->Q);
      float D_w = dot(normal_w, Q_w);
      vec3 w_new = n_w / dot(n_w, n_w);

      prim.quad.w = to_vec3f(w_new);
      prim.quad.normal = to_vec3f(normal_w);
      prim.quad.D = D_w;

      prim.material_id =
          get_or_add_material(quad_ptr->mat, linear_materials, linear_textures,
                              linear_perlin, image_buffer, mat_map, tex_map);
    }

    // Set AABB in World space using transform_bbox
    aabb box = current_trans.transform_bbox(node->bounding_box());

    // Pad zero-thickness AABBs to prevent GPU traversal misses
    double delta = 0.0001;
    if (box.x.max - box.x.min < delta) {
      box.x.min -= delta / 2.0;
      box.x.max += delta / 2.0;
    }
    if (box.y.max - box.y.min < delta) {
      box.y.min -= delta / 2.0;
      box.y.max += delta / 2.0;
    }
    if (box.z.max - box.z.min < delta) {
      box.z.min -= delta / 2.0;
      box.z.max += delta / 2.0;
    }

    linear_nodes[curr_idx].aabb_min =
        to_vec3f(vec3{box.x.min, box.y.min, box.z.min});
    linear_nodes[curr_idx].aabb_max =
        to_vec3f(vec3{box.x.max, box.y.max, box.z.max});

    linear_primitives.push_back(prim);

    linear_nodes[curr_idx].n_primitives = 1;
    linear_nodes[curr_idx].primitive_offset = prim_idx;

    return curr_idx;
  }
}
