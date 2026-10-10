#ifndef VEC3_HPP
#define VEC3_HPP

#include <cmath>
#include <iostream>

#include "cuda/vec.cuh" // RNG, shared with the device path
#include "rt.hpp"

class vec3 {
public:
  double e[3];
  vec3() : e{0, 0, 0} {}
  vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

  double x() const { return e[0]; }
  double y() const { return e[1]; }
  double z() const { return e[2]; }

  vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
  double operator[](int i) const { return e[i]; }
  double& operator[](int i) { return e[i]; }

  vec3& operator+=(const vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  vec3& operator*=(double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }

  vec3& operator/=(double t) { return *this *= 1 / t; }

  double length() const { return std::sqrt(length_squared()); }
  double length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }

  static vec3 random() {
    return vec3(random_double(), random_double(), random_double());
  }

  static vec3 random(double min, double max) {
    return vec3(random_double(min, max), random_double(min, max),
                random_double(min, max));
  }

  bool near_zero() const {
    // Return true if vector is close to zero in all dimensions
    auto s = 1e-8;
    return (std::fabs(e[0]) < s && std::fabs(e[1]) < s && std::fabs(e[2]) < s);
  }
};

// point3 is just an alias for vec3, but useful for geometric clarity
using point3 = vec3;

// Vector Utiliy Functions
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
  return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3& u, const vec3& v) {
  return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator+(const vec3& v, double t) {
  return vec3(v.e[0] + t, v.e[1] + t, v.e[2] + t);
}
inline vec3 operator+(double t, const vec3& v) { return v + t; }

inline vec3 operator-(const vec3& u, const vec3& v) {
  return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator-(const vec3& v, double t) {
  return vec3(v.e[0] - t, v.e[1] - t, v.e[2] - t);
}

inline vec3 operator*(const vec3& u, const vec3& v) {
  return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(const vec3& v, double t) {
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline vec3 operator*(double t, const vec3& v) { return v * t; }

inline vec3 operator/(const vec3& v, double t) { return (1 / t) * v; }

inline double dot(const vec3& u, const vec3& v) {
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3& u, const vec3& v) {
  return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
              u.e[2] * v.e[0] - u.e[0] * v.e[2],
              u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3 unit_vector(const vec3& v) { return v / v.length(); }

inline vec3 random_in_unit_sphere() {
  while (true) {
    auto p = vec3::random(-1, 1);
    if (p.length_squared() < 1) {
      return p;
    }
  }
}

inline vec3 random_in_unit_disk() {
  while (true) {
    auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
    if (p.length_squared() < 1) {
      return p;
    }
  }
}

inline vec3 random_unit_vector() {
  while (true) {
    auto p = vec3::random(-1, 1);
    auto lensq = p.length_squared();
    if (1e-160 < lensq && lensq <= 1) {
      return p / sqrt(lensq);
    }
  }
}

inline vec3 random_on_hemisphere(const vec3& normal) {
  vec3 on_unit_sphere = random_unit_vector();
  if (dot(on_unit_sphere, normal) > 0) { // In the same hemisphere as the normal
    return on_unit_sphere;
  } else {
    return -on_unit_sphere;
  }
}

// float64 mirrors of the device samplers in inc/cuda/vec.cuh. Same RNG, same
// draw count and order; only the arithmetic width differs. next_d() and
// next_f() return the same value -- bits/2^24 is exact in both formats -- so
// the random inputs are identical and any delta is pure precision.
inline vec3 random_unit_vector_d(RNG& rng) {
  double z = 1.0 - 2.0 * rng.next_d();
  double phi = 6.28318530717958648 * rng.next_d();
  double r = std::sqrt(std::fmax(0.0, (1.0 - z) * (1.0 + z)));
  return vec3(r * std::cos(phi), r * std::sin(phi), z);
}

inline vec3 random_in_unit_disk_d(RNG& rng) {
  double r = std::sqrt(rng.next_d());
  double theta = 6.28318530717958648 * rng.next_d();
  return vec3(r * std::cos(theta), r * std::sin(theta), 0.0);
}

inline vec3 reflect(const vec3& v, const vec3& n) {
  return v - (2 * dot(v, n) * n);
}

inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
  auto cos_theta = std::fmin(dot(-uv, n), 1.0);
  vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
  // Mirrors the device formula in inc/cuda/vec.cuh; see the note there for the
  // conditioning near the critical angle.
  double sin_theta = std::sqrt(std::fmax(0.0, (1.0 - cos_theta) * (1.0 + cos_theta)));
  double w = etai_over_etat * sin_theta;
  double cos_t = std::sqrt(std::fmax(0.0, (1.0 - w) * (1.0 + w)));
  return r_out_perp - cos_t * n;
}
#endif // !VEC3_HPP
