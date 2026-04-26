#pragma once
#include "cuda_structs.hpp"
#include "ray.cuh"
#include "vec.cuh"

__device__ inline vec3_gpu make_vec3_gpu(const Vec3f& v) {
  return vec3_gpu{v.x, v.y, v.z};
}

__device__ inline bool aabb_hit(const Vec3f& aabb_min, const Vec3f& aabb_max,
                                const ray_gpu& r, float t_min, float t_max) {
  for (int a = 0; a < 3; a++) {
    // Evaluate slab intersections per axis
    float invD = 1.0f / r.direction()[a];
    float t0 = (((float*)&aabb_min)[a] - r.origin()[a]) * invD;
    float t1 = (((float*)&aabb_max)[a] - r.origin()[a]) * invD;

    if (invD < 0.0f) {
      float temp = t0;
      t0 = t1;
      t1 = temp;
    }
    t_min = t0 > t_min ? t0 : t_min;
    t_max = t1 < t_max ? t1 : t_max;

    if (t_max <= t_min) return false;
  }
  return true;
}
// 2. The Union Unpacker & Intersection Logic
__device__ inline bool hit_primitive(const PrimitiveGPU& prim, const ray_gpu& r,
                                     float t_min, float t_max,
                                     HitRecordGPU& rec,
                                     curandState* local_rand_state) {

  if (prim.type == PrimitiveType::SPHERE) {
    // --- Unpack & Math Sphere ---
    vec3_gpu center = make_vec3_gpu(prim.sphere.center);
    float radius = prim.sphere.radius;

    vec3_gpu oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrtf(discriminant);
    // Find nearest root
    float root = (-half_b - sqrtd) / a;
    if (root <= t_min || t_max <= root) {
      root = (-half_b + sqrtd) / a;
      if (root <= t_min || t_max <= root) return false;
    }
    rec.t = root;
    vec3_gpu p = r.at(root);
    rec.p = {p.x(), p.y(), p.z()}; // Convert back to Vec3f for struct storage

    vec3_gpu outward_normal = (p - center) / radius;
    rec.front_face = dot(r.direction(), outward_normal) < 0.0f;

    vec3_gpu normal = rec.front_face ? outward_normal : -outward_normal;
    rec.normal = {normal.x(), normal.y(), normal.z()};

    float theta = acosf(-outward_normal.y());
    float phi = atan2f(-outward_normal.z(), outward_normal.x()) + 3.14159265f;
    rec.u = phi / (2.0f * 3.14159265f);
    rec.v = theta / 3.14159265f;
    rec.material_id = prim.material_id;
    return true;
  } else if (prim.type == PrimitiveType::MOVING_SPHERE) {
    // --- Unpack & Math Moving Sphere ---
    vec3_gpu center_start = make_vec3_gpu(prim.moving_sphere.center_start);
    vec3_gpu center_vec = make_vec3_gpu(prim.moving_sphere.center_vec);
    vec3_gpu center = center_start + center_vec * r.time();
    float radius = prim.moving_sphere.radius;

    vec3_gpu oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrtf(discriminant);
    // Find nearest root
    float root = (-half_b - sqrtd) / a;
    if (root <= t_min || t_max <= root) {
      root = (-half_b + sqrtd) / a;
      if (root <= t_min || t_max <= root) return false;
    }
    rec.t = root;
    vec3_gpu p = r.at(root);
    rec.p = {p.x(), p.y(), p.z()}; // Convert back to Vec3f for struct storage

    vec3_gpu outward_normal = (p - center) / radius;
    rec.front_face = dot(r.direction(), outward_normal) < 0.0f;

    vec3_gpu normal = rec.front_face ? outward_normal : -outward_normal;
    rec.normal = {normal.x(), normal.y(), normal.z()};

    float theta = acosf(-outward_normal.y());
    float phi = atan2f(-outward_normal.z(), outward_normal.x()) + 3.14159265f;
    rec.u = phi / (2.0f * 3.14159265f);
    rec.v = theta / 3.14159265f;
    rec.material_id = prim.material_id;
    return true;
  } else if (prim.type == PrimitiveType::QUAD) {
    // --- Unpack & Math Quad ---
    vec3_gpu Q = make_vec3_gpu(prim.quad.Q);
    vec3_gpu u = make_vec3_gpu(prim.quad.u);
    vec3_gpu v = make_vec3_gpu(prim.quad.v);
    vec3_gpu w = make_vec3_gpu(prim.quad.w);
    vec3_gpu quad_normal = make_vec3_gpu(prim.quad.normal);
    float D = prim.quad.D;
    float denom = dot(quad_normal, r.direction());
    // Ray is parallel to plane
    if (fabsf(denom) < 1e-8) return false;
    float t = (D - dot(quad_normal, r.origin())) / denom;
    if (t <= t_min || t >= t_max) return false;
    vec3_gpu p = r.at(t);

    // Planar intersection coordinates
    vec3_gpu planar_hitpt = p - Q;
    float alpha = dot(w, cross(planar_hitpt, v));
    float beta = dot(w, cross(u, planar_hitpt));
    // Is the hit point outside the Quad?
    if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1) return false;
    rec.t = t;
    rec.p = {p.x(), p.y(), p.z()};
    rec.u = alpha;
    rec.v = beta;

    rec.front_face = dot(r.direction(), quad_normal) < 0.0f;
    vec3_gpu normal = rec.front_face ? quad_normal : -quad_normal;
    rec.normal = {normal.x(), normal.y(), normal.z()};
    rec.material_id = prim.material_id;
    return true;

  } else if (prim.type == PrimitiveType::VOLUME_SPHERE) {
    vec3_gpu center = make_vec3_gpu(prim.volume_sphere.center);
    float radius = prim.volume_sphere.radius;

    vec3_gpu oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) return false;

    float sqrtd = sqrtf(discriminant);
    float root1 = (-half_b - sqrtd) / a;
    float root2 = (-half_b + sqrtd) / a;

    float t1 = fmaxf(root1, t_min);
    float t2 = fminf(root2, t_max);

    if (t1 >= t2) return false;
    if (t1 < 0.0f) t1 = 0.0f;

    float ray_length = r.direction().length();
    float distance_inside_boundary = (t2 - t1) * ray_length;
    float hit_distance = prim.volume_sphere.neg_inv_density *
                         logf(curand_uniform(local_rand_state));

    if (hit_distance > distance_inside_boundary) return false;

    rec.t = t1 + hit_distance / ray_length;
    vec3_gpu p = r.at(rec.t);
    rec.p = {p.x(), p.y(), p.z()};
    rec.normal = {1.0f, 0.0f, 0.0f}; // arbitrary
    rec.front_face = true;
    rec.u = 0.0f;
    rec.v = 0.0f;
    rec.material_id = prim.material_id;
    return true;

  } else if (prim.type == PrimitiveType::VOLUME_BOX) {
    vec3_gpu r_o = r.origin() - make_vec3_gpu(prim.volume_box.offset);
    float s_theta = prim.volume_box.sin_theta;
    float c_theta = prim.volume_box.cos_theta;

    vec3_gpu origin(c_theta * r_o.x() - s_theta * r_o.z(), r_o.y(),
                    s_theta * r_o.x() + c_theta * r_o.z());
    vec3_gpu r_d = r.direction();
    vec3_gpu dir(c_theta * r_d.x() - s_theta * r_d.z(), r_d.y(),
                 s_theta * r_d.x() + c_theta * r_d.z());
    ray_gpu local_ray(origin, dir, r.time());

    Vec3f local_min = prim.volume_box.local_min;
    Vec3f local_max = prim.volume_box.local_max;

    float t0_overall = -99999.0f;
    float t1_overall = 99999.0f;

    for (int a = 0; a < 3; a++) {
      float invD = 1.0f / local_ray.direction()[a];
      float t0 = (((float*)&local_min)[a] - local_ray.origin()[a]) * invD;
      float t1 = (((float*)&local_max)[a] - local_ray.origin()[a]) * invD;
      if (invD < 0.0f) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
      }
      t0_overall = t0 > t0_overall ? t0 : t0_overall;
      t1_overall = t1 < t1_overall ? t1 : t1_overall;
      if (t1_overall <= t0_overall) return false;
    }

    float t1 = fmaxf(t0_overall, t_min);
    float t2 = fminf(t1_overall, t_max);

    if (t1 >= t2) return false;
    if (t1 < 0.0f) t1 = 0.0f;

    float ray_length = r.direction().length();
    float distance_inside_boundary = (t2 - t1) * ray_length;
    float hit_distance = prim.volume_box.neg_inv_density *
                         logf(curand_uniform(local_rand_state));

    if (hit_distance > distance_inside_boundary) return false;

    rec.t = t1 + hit_distance / ray_length;
    vec3_gpu p = r.at(rec.t);
    rec.p = {p.x(), p.y(), p.z()};
    rec.normal = {1.0f, 0.0f, 0.0f}; // arbitrary
    rec.front_face = true;
    rec.u = 0.0f;
    rec.v = 0.0f;
    rec.material_id = prim.material_id;
    return true;
  }
  return false;
}

__device__ inline bool hit_linear_bvh(const LinearBVHNode* bvh_nodes,
                                      const PrimitiveGPU* primitives,
                                      const ray_gpu& ray, float t_min,
                                      float t_max, HitRecordGPU& rec,
                                      curandState* local_rand_state) {
  int stack[64];
  int stack_ptr = 0;

  stack[stack_ptr++] = 0;
  bool hit_anything = false;
  float closest_so_far = t_max;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    const LinearBVHNode& node = bvh_nodes[node_idx];

    if (!aabb_hit(node.aabb_min, node.aabb_max, ray, t_min, closest_so_far)) {
      continue;
    }

    if (node.n_primitives > 0) { // Leaf
      HitRecordGPU temp_rec;
      const PrimitiveGPU& prim = primitives[node.primitive_offset];

      if (hit_primitive(prim, ray, t_min, closest_so_far, temp_rec,
                        local_rand_state)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    } else { // Interior
      // Push right first so left will be processed first becuase of LIFO
      stack[stack_ptr++] = node.second_child_offset;
      stack[stack_ptr++] = node_idx + 1;
    }
  }

  return hit_anything;
}
