#ifndef RT_HPP
#define RT_HPP

#include <limits>
#include <random>

// Constants

const double infinity = std::numeric_limits<double>::infinity();

// Utility Functions

inline double degrees_to_radians(double degrees) { return degrees * std::numbers::pi / 180.0; }

// Scene-construction RNG only. Not thread-safe, and deliberately NOT used by
// either render path -- both now draw from the per-path counter-based RNG in
// inc/cuda/vec.cuh. Calling this from render threads would reintroduce a data
// race on the shared mt19937 state.
inline std::mt19937& scene_generator() {
  static std::mt19937 generator;
  return generator;
}

// Reset before building a scene so geometry is reproducible regardless of how
// many scenes were built earlier in the process.
inline void seed_scene_rng(std::mt19937::result_type seed) { scene_generator().seed(seed); }

inline double random_double() {
  // Returns a random real in [0,1)
  static std::uniform_real_distribution<double> distribution(0.0, 1.0);
  return distribution(scene_generator());
}

inline double random_double(double min, double max) {
  // Returns a random real in [min, max]
  return min + (max - min) * random_double();
}

inline int random_int(int min, int max) {
  // Returns a random integer in [min,max].
  return int(random_double(min, max + 1));
}

#endif // !RT_HPP
