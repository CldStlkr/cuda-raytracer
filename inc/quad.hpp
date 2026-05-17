#ifndef QUAD_HPP
#define QUAD_HPP

#include "hittable_list.hpp"
#include "material.hpp"
#include "vec3.hpp"
#include <memory>

class quad : public hittable {
public:
  quad(const point3& Q, const vec3& u, const vec3& v,
       std::shared_ptr<material> mat)
      : Q{Q}, u{u}, v{v}, mat{mat} {
    auto n = cross(u, v);
    normal = unit_vector(n);
    D = dot(normal, Q);
    w = n / dot(n, n);

    set_bounding_box();
  }

  virtual void set_bounding_box() {
    // Compute the bounding box of all four verticies
    auto bbox_diagonal1 = aabb(Q, Q + u + v);
    auto bbox_diagonal2 = aabb(Q + u, Q + v);
    bbox = aabb(bbox_diagonal1, bbox_diagonal2);
  }

  aabb bounding_box() const override { return bbox; }
  bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
    auto denom = dot(normal, r.direction());

    // No hit if the ray is parallel to the plane;
    if (std::fabs(denom) < 1e-8) {
      return false;
    }

    // Return false if the hit point parameter t is outside the ray interval
    auto t = (D - dot(normal, r.origin())) / denom;
    if (!ray_t.contains(t)) {
      return false;
    }

    // Determine if the hit point lies within the planar shape using its plane
    // coordinates.
    auto intersection = r.at(t);
    vec3 planar_hitpt_vector = intersection - Q;
    auto alpha = dot(w, cross(planar_hitpt_vector, v));
    auto beta = dot(w, cross(u, planar_hitpt_vector));

    if (!is_interior(alpha, beta, rec)) return false;

    // Ray hits the 2D shape; set the rest of the hit record and return true.

    rec.t = t;
    rec.p = intersection;
    rec.mat = mat;
    rec.set_face_normal(r, normal);

    return true;
  }

  virtual bool is_interior(double a, double b, hit_record& rec) const {
    interval unit_interval = interval(0, 1);
    // Given the hit point in plane coordinates, return false if it is outside
    // the primitive, otherwise set the hit record UV coordinates and return
    // true.

    if (!unit_interval.contains(a) || !unit_interval.contains(b)) return false;

    rec.u = a;
    rec.v = b;
    return true;
  }

  point3 Q;
  vec3 u, v;
  vec3 w;
  std::shared_ptr<material> mat;
  aabb bbox;
  vec3 normal;
  double D;
};

class moving_quad : public hittable {
public:
  moving_quad(const point3& Q1, const point3& Q2, const vec3& u, const vec3& v,
              std::shared_ptr<material> mat)
      : Q1(Q1), Q2(Q2), u(u), v(v), mat(mat) {
    auto n = cross(u, v);
    normal = unit_vector(n);
    D1 = dot(normal, Q1);
    D2 = dot(normal, Q2);
    w = n / dot(n, n);
    set_bounding_box();
  }

  void set_bounding_box() {
    auto bbox1 = aabb(aabb(Q1, Q1 + u + v), aabb(Q1 + u, Q1 + v));
    auto bbox2 = aabb(aabb(Q2, Q2 + u + v), aabb(Q2 + u, Q2 + v));
    bbox = aabb(bbox1, bbox2);
  }

  aabb bounding_box() const override { return bbox; }

  bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
    point3 Q = Q1 + (Q2 - Q1) * r.time();
    double D = D1 + (D2 - D1) * r.time();

    auto denom = dot(normal, r.direction());
    if (std::fabs(denom) < 1e-8) return false;
    auto t = (D - dot(normal, r.origin())) / denom;
    if (!ray_t.contains(t)) return false;

    auto intersection = r.at(t);
    vec3 planar_hitpt_vector = intersection - Q;
    auto alpha = dot(w, cross(planar_hitpt_vector, v));
    auto beta = dot(w, cross(u, planar_hitpt_vector));

    if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1) return false;

    rec.t = t;
    rec.p = intersection;
    rec.mat = mat;
    rec.set_face_normal(r, normal);
    rec.u = alpha;
    rec.v = beta;
    return true;
  }

  std::shared_ptr<material> get_material() const { return mat; }

  point3 Q1, Q2;
  vec3 u, v, w;
  std::shared_ptr<material> mat;
  aabb bbox;
  vec3 normal;
  double D1, D2;
};

inline shared_ptr<hittable_list> moving_box(const point3& a1, const point3& a2,
                                            const point3& b1, const point3& b2,
                                            shared_ptr<material> mat) {
  // Returns a 3D box that moves from [a1, b1] to [a2, b2]
  auto sides = std::make_shared<hittable_list>();

  auto min1 = point3(std::fmin(a1.x(), b1.x()), std::fmin(a1.y(), b1.y()),
                     std::fmin(a1.z(), b1.z()));
  auto max1 = point3(std::fmax(a1.x(), b1.x()), std::fmax(a1.y(), b1.y()),
                     std::fmax(a1.z(), b1.z()));

  auto min2 = point3(std::fmin(a2.x(), b2.x()), std::fmin(a2.y(), b2.y()),
                     std::fmin(a2.z(), b2.z()));
  auto max2 = point3(std::fmax(a2.x(), b2.x()), std::fmax(a2.y(), b2.y()),
                     std::fmax(a2.z(), b2.z()));

  auto dx1 = vec3(max1.x() - min1.x(), 0, 0);
  auto dy1 = vec3(0, max1.y() - min1.y(), 0);
  auto dz1 = vec3(0, 0, max1.z() - min1.z());

  [[maybe_unused]] auto dx2 = vec3(max2.x() - min2.x(), 0, 0);
  [[maybe_unused]] auto dy2 = vec3(0, max2.y() - min2.y(), 0);
  auto dz2 = vec3(0, 0, max2.z() - min2.z());

  // Assume u, v vectors are constant for "vibration" (just Q changes)
  // For a general moving box where size changes, it's more complex.
  // But for the user request "vibration", we just move the origin.

  sides->add(std::make_shared<moving_quad>(point3(min1.x(), min1.y(), max1.z()),
                                           point3(min2.x(), min2.y(), max2.z()),
                                           dx1, dy1, mat)); // front
  sides->add(std::make_shared<moving_quad>(point3(max1.x(), min1.y(), max1.z()),
                                           point3(max2.x(), min2.y(), max2.z()),
                                           -dz1, dy1, mat)); // right
  sides->add(std::make_shared<moving_quad>(
      point3(max1.x(), min1.y(), max1.z() - dz1.z()),
      point3(max2.x(), min2.y(), max2.z() - dz2.z()), -dx1, dy1, mat)); // back
  sides->add(std::make_shared<moving_quad>(point3(min1.x(), min1.y(), min1.z()),
                                           point3(min2.x(), min2.y(), min2.z()),
                                           dz1, dy1, mat)); // left
  sides->add(std::make_shared<moving_quad>(point3(min1.x(), max1.y(), max1.z()),
                                           point3(min2.x(), max2.y(), max2.z()),
                                           dx1, -dz1, mat)); // top
  sides->add(std::make_shared<moving_quad>(point3(min1.x(), min1.y(), min1.z()),
                                           point3(min2.x(), min2.y(), min2.z()),
                                           dx1, dz1, mat)); // bottom

  return sides;
}

inline shared_ptr<hittable_list> box(const point3& a, const point3& b,
                                     shared_ptr<material> mat) {
  // Returns the 3D box (six sides) that contains the two opposite vertices a &
  // b.

  auto sides = std::make_shared<hittable_list>();

  // Construct the two opposite vertices with the minimum and maximum
  // coordinates.
  auto min = point3(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()),
                    std::fmin(a.z(), b.z()));
  auto max = point3(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()),
                    std::fmax(a.z(), b.z()));

  auto dx = vec3(max.x() - min.x(), 0, 0);
  auto dy = vec3(0, max.y() - min.y(), 0);
  auto dz = vec3(0, 0, max.z() - min.z());

  sides->add(std::make_shared<quad>(point3(min.x(), min.y(), max.z()), dx, dy,
                                    mat)); // front
  sides->add(std::make_shared<quad>(point3(max.x(), min.y(), max.z()), -dz, dy,
                                    mat)); // right
  sides->add(std::make_shared<quad>(point3(max.x(), min.y(), min.z()), -dx, dy,
                                    mat)); // back
  sides->add(std::make_shared<quad>(point3(min.x(), min.y(), min.z()), dz, dy,
                                    mat)); // left
  sides->add(std::make_shared<quad>(point3(min.x(), max.y(), max.z()), dx, -dz,
                                    mat)); // top
  sides->add(std::make_shared<quad>(point3(min.x(), min.y(), min.z()), dx, dz,
                                    mat)); // bottom

  return sides;
}

#endif
