#pragma once
#include "cuda_structs.hpp"
#include "ray.cuh"
#include "vec.cuh"

// Returns t_near of the ray-AABB intersection, or 1e30f on miss.
// inv_dir must be precomputed as 1/ray.direction() per component.
__device__ inline float aabb_hit_tmin(const vec3_gpu& aabb_min, const vec3_gpu& aabb_max, const vec3_gpu& inv_dir,
                                      const vec3_gpu& ray_orig, float t_min, float t_max) {
  float t0 = (aabb_min.x() - ray_orig.x()) * inv_dir.x();
  float t1 = (aabb_max.x() - ray_orig.x()) * inv_dir.x();
  t_min = fmaxf(t_min, fminf(t0, t1));
  t_max = fminf(t_max, fmaxf(t0, t1));

  t0 = (aabb_min.y() - ray_orig.y()) * inv_dir.y();
  t1 = (aabb_max.y() - ray_orig.y()) * inv_dir.y();
  t_min = fmaxf(t_min, fminf(t0, t1));
  t_max = fminf(t_max, fmaxf(t0, t1));

  t0 = (aabb_min.z() - ray_orig.z()) * inv_dir.z();
  t1 = (aabb_max.z() - ray_orig.z()) * inv_dir.z();
  t_min = fmaxf(t_min, fminf(t0, t1));
  t_max = fminf(t_max, fmaxf(t0, t1));

  return (t_max > t_min) ? t_min : 1e30f;
}

__device__ inline bool hit_primitive(const PrimitiveGPU& prim, const ray_gpu& r, float t_min, float t_max,
                                     HitRecordGPU& rec, RNG& rng) {

  if (prim.type == PrimitiveType::SPHERE) {
    vec3_gpu center = prim.sphere.center;
    float radius = prim.sphere.radius;

    vec3_gpu oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root <= t_min || t_max <= root) {
      root = (-half_b + sqrtd) / a;
      if (root <= t_min || t_max <= root) return false;
    }
    rec.t = root;
    vec3_gpu p = r.at(root);
    rec.p = p;

    vec3_gpu outward_normal = (p - center) / radius;
    rec.front_face = dot(r.direction(), outward_normal) < 0.0f;

    vec3_gpu normal = rec.front_face ? outward_normal : -outward_normal;
    rec.normal = normal;

    float theta = acosf(-outward_normal.y());
    float phi = atan2f(-outward_normal.z(), outward_normal.x()) + 3.14159265f;
    rec.u = phi / (2.0f * 3.14159265f);
    rec.v = theta / 3.14159265f;
    rec.material_id = prim.material_id;
    return true;
  } else if (prim.type == PrimitiveType::MOVING_SPHERE) {
    vec3_gpu center_start = prim.moving_sphere.center_start;
    vec3_gpu center_vec = prim.moving_sphere.center_vec;
    vec3_gpu center = center_start + center_vec * r.time();
    float radius = prim.moving_sphere.radius;

    vec3_gpu oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root <= t_min || t_max <= root) {
      root = (-half_b + sqrtd) / a;
      if (root <= t_min || t_max <= root) return false;
    }
    rec.t = root;
    vec3_gpu p = r.at(root);
    rec.p = p;

    vec3_gpu outward_normal = (p - center) / radius;
    rec.front_face = dot(r.direction(), outward_normal) < 0.0f;

    vec3_gpu normal = rec.front_face ? outward_normal : -outward_normal;
    rec.normal = normal;

    float theta = acosf(-outward_normal.y());
    float phi = atan2f(-outward_normal.z(), outward_normal.x()) + 3.14159265f;
    rec.u = phi / (2.0f * 3.14159265f);
    rec.v = theta / 3.14159265f;
    rec.material_id = prim.material_id;
    return true;
  } else if (prim.type == PrimitiveType::QUAD) {
    vec3_gpu Q = prim.quad.Q;
    vec3_gpu u = prim.quad.u;
    vec3_gpu v = prim.quad.v;
    vec3_gpu w = prim.quad.w;
    vec3_gpu quad_normal = prim.quad.normal;
    float D = prim.quad.D;
    float denom = dot(quad_normal, r.direction());
    if (fabsf(denom) < 1e-8) return false;
    float t = (D - dot(quad_normal, r.origin())) / denom;
    if (t <= t_min || t >= t_max) return false;
    vec3_gpu p = r.at(t);

    vec3_gpu planar_hitpt = p - Q;
    float alpha = dot(w, cross(planar_hitpt, v));
    float beta = dot(w, cross(u, planar_hitpt));
    if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1) return false;
    rec.t = t;
    rec.p = p;
    rec.u = alpha;
    rec.v = beta;

    rec.front_face = dot(r.direction(), quad_normal) < 0.0f;
    vec3_gpu normal = rec.front_face ? quad_normal : -quad_normal;
    rec.normal = normal;
    rec.material_id = prim.material_id;
    return true;

  } else if (prim.type == PrimitiveType::VOLUME_SPHERE) {
    vec3_gpu center = prim.volume_sphere.center;
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
    float hit_distance = prim.volume_sphere.neg_inv_density * logf(rng.next_f());

    if (hit_distance > distance_inside_boundary) return false;

    rec.t = t1 + hit_distance / ray_length;
    vec3_gpu p = r.at(rec.t);
    rec.p = p;
    rec.normal = vec3_gpu(1.0f, 0.0f, 0.0f);
    rec.front_face = true;
    rec.u = 0.0f;
    rec.v = 0.0f;
    rec.material_id = prim.material_id;
    return true;

  } else if (prim.type == PrimitiveType::MOVING_QUAD) {
    vec3_gpu Q_start = prim.moving_quad.Q_start;
    vec3_gpu Q_vec = prim.moving_quad.Q_vec;
    vec3_gpu Q_current = Q_start + Q_vec * r.time();

    float D_current = prim.moving_quad.D_start + prim.moving_quad.D_vec * r.time();
    vec3_gpu normal = prim.moving_quad.normal;
    vec3_gpu w = prim.moving_quad.w;

    float denom = dot(normal, r.direction());
    if (fabsf(denom) < 1e-8f) return false;

    float t = (D_current - dot(normal, r.origin())) / denom;
    if (t < t_min || t > t_max) return false;

    vec3_gpu intersection = r.at(t);
    vec3_gpu planar_hitpt_vector = intersection - Q_current;
    float alpha = dot(w, cross(planar_hitpt_vector, prim.moving_quad.v));
    float beta = dot(w, cross(prim.moving_quad.u, planar_hitpt_vector));

    if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) return false;

    rec.t = t;
    rec.p = intersection;
    rec.material_id = prim.material_id;
    rec.u = alpha;
    rec.v = beta;
    rec.front_face = dot(r.direction(), normal) < 0.0f;
    rec.normal = rec.front_face ? normal : -normal;
    return true;

  } else if (prim.type == PrimitiveType::VOLUME_BOX) {
    vec3_gpu r_o = r.origin() - prim.volume_box.offset;
    float s_theta = prim.volume_box.sin_theta;
    float c_theta = prim.volume_box.cos_theta;

    vec3_gpu origin(c_theta * r_o.x() - s_theta * r_o.z(), r_o.y(), s_theta * r_o.x() + c_theta * r_o.z());
    vec3_gpu r_d = r.direction();
    vec3_gpu dir(c_theta * r_d.x() - s_theta * r_d.z(), r_d.y(), s_theta * r_d.x() + c_theta * r_d.z());
    ray_gpu local_ray(origin, dir, r.time());

    vec3_gpu local_min = prim.volume_box.local_min;
    vec3_gpu local_max = prim.volume_box.local_max;

    float t0_overall = -99999.0f;
    float t1_overall = 99999.0f;

    for (int a = 0; a < 3; a++) {
      float invD = 1.0f / local_ray.direction()[a];
      float t0 = (local_min[a] - local_ray.origin()[a]) * invD;
      float t1 = (local_max[a] - local_ray.origin()[a]) * invD;
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
    float hit_distance = prim.volume_box.neg_inv_density * logf(rng.next_f());

    if (hit_distance > distance_inside_boundary) return false;

    rec.t = t1 + hit_distance / ray_length;
    rec.p = r.at(rec.t);
    rec.normal = vec3_gpu(1.0f, 0.0f, 0.0f);
    rec.front_face = true;
    rec.u = 0.0f;
    rec.v = 0.0f;
    rec.material_id = prim.material_id;
    return true;
  }
  return false;
}

__device__ inline bool hit_linear_bvh(const cuda::span<LinearBVHNode> bvh_nodes,
                                      const cuda::span<PrimitiveGPU> primitives, const ray_gpu& ray, float t_min,
                                      float t_max, HitRecordGPU& rec, RNG& rng) {
  vec3_gpu inv_dir(1.0f / ray.direction().x(), 1.0f / ray.direction().y(), 1.0f / ray.direction().z());
  vec3_gpu ray_orig = ray.origin();

  int stack[64];
  int stack_ptr = 0;
  bool hit_anything = false;
  float closest_so_far = t_max;

  // Early exit if root AABB misses
  if (aabb_hit_tmin(bvh_nodes[0].aabb_min, bvh_nodes[0].aabb_max, inv_dir, ray_orig, t_min, closest_so_far) >= 1e30f)
    return false;
  stack[stack_ptr++] = 0;

  while (stack_ptr > 0) {
    int node_idx = stack[--stack_ptr];
    const LinearBVHNode& node = bvh_nodes[node_idx];

    if (node.n_primitives > 0) { // Leaf
      HitRecordGPU temp_rec;
      if (hit_primitive(primitives[node.primitive_offset], ray, t_min, closest_so_far, temp_rec, rng)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    } else { // Interior: test both children, push misses, process nearer first
      int left = node_idx + 1;
      int right = node.second_child_offset;

      float t_left =
          aabb_hit_tmin(bvh_nodes[left].aabb_min, bvh_nodes[left].aabb_max, inv_dir, ray_orig, t_min, closest_so_far);
      float t_right =
          aabb_hit_tmin(bvh_nodes[right].aabb_min, bvh_nodes[right].aabb_max, inv_dir, ray_orig, t_min, closest_so_far);

      // Push farther child first so nearer child is popped first (LIFO)
      if (t_left > t_right) {
        if (t_left < 1e30f) stack[stack_ptr++] = left;
        if (t_right < 1e30f) stack[stack_ptr++] = right;
      } else {
        if (t_right < 1e30f) stack[stack_ptr++] = right;
        if (t_left < 1e30f) stack[stack_ptr++] = left;
      }
    }
  }

  return hit_anything;
}
