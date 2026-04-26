#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "color.hpp"
#include "hittable.hpp"
#include "texture.hpp"
#include <memory>

class material {
public:
  virtual ~material() = default;

  virtual color emitted(double /*u*/, double /*v*/,
                        const point3 & /*p*/) const {
    return color(0, 0, 0);
  }
  virtual bool scatter(const ray & /*r_in*/, const hit_record & /*rec*/,
                       color & /*attenuation*/, ray & /*scattered*/) const {
    return false;
  }
};

class lambertian : public material {
public:
  lambertian(const color &albedo)
      : tex{std::make_shared<solid_color>(albedo)} {}
  lambertian(std::shared_ptr<texture> tex) : tex{tex} {}

  bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
               ray &scattered) const override {
    auto scatter_direction = rec.normal + random_unit_vector();

    if (scatter_direction.near_zero()) {
      scatter_direction = rec.normal;
    }

    scattered = ray(rec.p, scatter_direction, r_in.time());
    attenuation = tex->value(rec.u, rec.v, rec.p);
    return true;
  }

  color get_albedo() const {
    // Book 2 delegates storage to the texture pointer.
    // We retrieve the flat albedo for GPU static mapping via value():
    return tex->value(0.0, 0.0, point3(0, 0, 0));
  }

public:
  std::shared_ptr<texture> tex;
};

class metal : public material {
public:
  metal(const color &albedo, double fuzz)
      : albedo{albedo}, fuzz{fuzz < 1 ? fuzz : 1} {}

  bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
               ray &scattered) const override {
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (fuzz * random_unit_vector());
    scattered = ray(rec.p, reflected, r_in.time());
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }

  color get_albedo() const { return albedo; }
  double get_fuzz() const { return fuzz; }

private:
  color albedo;
  double fuzz;
};

class dielectric : public material {
public:
  dielectric(double refraction_index) : refraction_index{refraction_index} {}
  bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
               ray &scattered) const override {
    attenuation = color(1.0, 1.0, 1.0);
    double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

    vec3 unit_direction = unit_vector(r_in.direction());
    vec3 refracted = refract(unit_direction, rec.normal, ri);

    scattered = ray(rec.p, refracted, r_in.time());

    double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    vec3 direction;

    if (cannot_refract || reflectance(cos_theta, ri) > random_double()) {
      direction = reflect(unit_direction, rec.normal);
    } else {
      direction = refract(unit_direction, rec.normal, ri);
    }

    return true;
  }

  double get_refraction_index() const { return refraction_index; }

private:
  // Refractive index in vacuum or air, or the ratio of the material's
  // refractive index over the refractive index of the enclosing media
  double refraction_index;

  static double reflectance(double cosine, double refraction_index) {
    // Use Schlick's approximation for reflectance
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow((1 - cosine), 5);
  }
};

class diffuse_light : public material {
public:
  diffuse_light(std::shared_ptr<texture> tex) : tex{tex} {}
  diffuse_light(const color &emit) : tex{std::make_shared<solid_color>(emit)} {}

  color emitted(double u, double v, const point3 &p) const override {
    return tex->value(u, v, p);
  }

public:
  std::shared_ptr<texture> tex;
};

class isotropic : public material {
public:
  isotropic(const color &albedo) : tex(std::make_shared<solid_color>(albedo)) {}
  isotropic(std::shared_ptr<texture> tex) : tex(tex) {}

  bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
               ray &scattered) const override {
    scattered = ray(rec.p, random_unit_vector(), r_in.time());
    attenuation = tex->value(rec.u, rec.v, rec.p);
    return true;
  }

public:
  std::shared_ptr<texture> tex;
};

#endif // !MATERIAL_HPP
