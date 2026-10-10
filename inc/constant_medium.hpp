#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "hittable.hpp"
#include "material.hpp"
#include "texture.hpp"

class constant_medium : public hittable {
public:
  constant_medium(std::shared_ptr<hittable> boundary, double density,
                  std::shared_ptr<texture> tex)
      : boundary(boundary), neg_inv_density(-1 / density),
        phase_function(make_shared<isotropic>(tex)) {}

  constant_medium(std::shared_ptr<hittable> boundary, double density,
                  const color &albedo)
      : boundary(boundary), neg_inv_density(-1 / density),
        phase_function(std::make_shared<isotropic>(albedo)) {}

  bool hit(const ray &r, interval ray_t, hit_record &rec, RNG &rng) const override {
    hit_record rec1, rec2;

    if (!boundary->hit(r, interval::universe, rec1, rng))
      return false;

    if (!boundary->hit(r, interval(rec1.t + 0.0001, infinity), rec2, rng))
      return false;

    if (rec1.t < ray_t.min)
      rec1.t = ray_t.min;
    if (rec2.t > ray_t.max)
      rec2.t = ray_t.max;

    if (rec1.t >= rec2.t)
      return false;

    if (rec1.t < 0)
      rec1.t = 0;

    auto ray_length = r.direction().length();
    auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    // 1 draw from the stream-1 RNG, matching hit_primitive's VOLUME_* cases.
    // NOTE: the number of draws per bounce is traversal-order dependent -- one
    // per volume reaching this point -- and the host tree walk and the device
    // flat-BVH walk do not visit volumes in the same order. Volume scenes
    // therefore cannot be image-compared against the device; see validation
    // notes. Non-volume scenes consume zero draws here and are unaffected.
    auto hit_distance = neg_inv_density * std::log(rng.next_d());

    if (hit_distance > distance_inside_boundary)
      return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    rec.normal = vec3(1, 0, 0); // arbitrary
    rec.front_face = true;      // also arbitrary
    rec.mat = phase_function;

    return true;
  }

  aabb bounding_box() const override { return boundary->bounding_box(); }

public:
  std::shared_ptr<hittable> boundary;
  double neg_inv_density;
  std::shared_ptr<material> phase_function;
};

#endif
