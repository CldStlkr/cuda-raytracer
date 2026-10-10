#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "color.hpp"
#include "hittable.hpp"
#include "texture.hpp"
#include <memory>

class material {
public:
  virtual ~material() = default;

  virtual color emitted(double /*u*/, double /*v*/, const point3& /*p*/) const {
    return color(0, 0, 0);
  }
  // rng is seeded per (pixel, sample, bounce, stream=2) by the caller, matching
  // shade_kernel. Overrides must consume a fixed number of draws so the host
  // and device streams stay aligned.
  virtual bool scatter(const ray& /*r_in*/, const hit_record& /*rec*/,
                       color& /*attenuation*/, ray& /*scattered*/,
                       RNG& /*rng*/) const {
    return false;
  }
};

class lambertian : public material {
public:
  lambertian(const color& albedo)
      : tex{std::make_shared<solid_color>(albedo)} {}
  lambertian(std::shared_ptr<texture> tex) : tex{tex} {}

  bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
               ray& scattered, RNG& rng) const override {
    auto scatter_direction = rec.normal + random_unit_vector_d(rng); // 2 draws

    // length_squared(), not near_zero(): near_zero() tests each component
    // against 1e-8 (|v| < 1.7e-8) while shade_kernel tests |v|^2 < 1e-8
    // (|v| < 1e-4). Unreachable in practice with analytic sampling, but the
    // two paths must agree on the predicate.
    if (scatter_direction.length_squared() < 1e-8) {
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
  metal(const color& albedo, double fuzz)
      : albedo{albedo}, fuzz{fuzz < 1 ? fuzz : 1} {}

  bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
               ray& scattered, RNG& rng) const override {
    // Two deliberate changes to match shade_kernel: fuzz perturbs by a point
    // ON the unit sphere (was random_in_unit_sphere, i.e. inside it), and the
    // reflected vector is not re-normalized -- reflect() of a unit vector
    // about a unit normal is already unit, so unit_vector() only added a
    // division the device path doesn't perform.
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    reflected = reflected + (fuzz * random_unit_vector_d(rng)); // 2 draws
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
  bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
               ray& scattered, RNG& rng) const override {
    attenuation = color(1.0, 1.0, 1.0);
    double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

    vec3 unit_direction = unit_vector(r_in.direction());

    double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
    // Matches the device formula in cuda_renderer.cu so the two paths agree
    // bit-for-bit; see the note there.
    double sin_theta = std::sqrt(std::fmax(0.0, (1.0 - cos_theta) * (1.0 + cos_theta)));

    bool cannot_refract = ri * sin_theta > 1.0;
    vec3 direction;

    // Drawn unconditionally so the per-material draw count is a function of
    // material type alone, not of a floating-point comparison. Defensive
    // rather than load-bearing: this stream is re-seeded every bounce, so a
    // short-circuit here could not desync anything downstream today. It keeps
    // the count an invariant the parity test can assert.
    double xi = rng.next_d(); // 1 draw, always

    if (cannot_refract || reflectance(cos_theta, ri) > xi) {
      direction = reflect(unit_direction, rec.normal);
    } else {
      direction = refract(unit_direction, rec.normal, ri);
    }

    scattered = ray(rec.p, direction, r_in.time());

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
    // x^5 by multiplication, matching reflectance() in inc/cuda/vec.cuh so the
    // two paths evaluate the same expression.
    double x = 1 - cosine, x2 = x * x;
    return r0 + (1 - r0) * (x2 * x2 * x);
  }
};

class diffuse_light : public material {
public:
  diffuse_light(std::shared_ptr<texture> tex) : tex{tex} {}
  diffuse_light(const color& emit) : tex{std::make_shared<solid_color>(emit)} {}

  color emitted(double u, double v, const point3& p) const override {
    return tex->value(u, v, p);
  }

public:
  std::shared_ptr<texture> tex;
};

class isotropic : public material {
public:
  isotropic(const color& albedo) : tex(std::make_shared<solid_color>(albedo)) {}
  isotropic(std::shared_ptr<texture> tex) : tex(tex) {}

  bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
               ray& scattered, RNG& rng) const override {
    scattered = ray(rec.p, random_unit_vector_d(rng), r_in.time()); // 2 draws
    attenuation = tex->value(rec.u, rec.v, rec.p);
    return true;
  }

public:
  std::shared_ptr<texture> tex;
};

#endif // !MATERIAL_HPP
