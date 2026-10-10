#include "vulkan_app.hpp"

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "With no options, launches the interactive GUI.\n\n"
      << "  --headless            render without a window and exit\n"
      << "  --scene <name>        " << scene_name_list() << " (default: static)\n"
      << "  --width <px>          image width; height follows the scene aspect (default: 800)\n"
      << "  --spp <n>             samples per pixel (default: 10)\n"
      << "  --depth <n>           maximum path depth (default: 10)\n"
      << "  --backend <b>         gpu | cpu | both (default: gpu)\n"
      << "  --out <prefix>        output prefix; writes <prefix>_gpu.ppm / _cpu.ppm (default: render)\n"
      << "  -h, --help            this message\n\n"
      << "cpu renders the float64 reference, gpu the float32 kernels. Both share one\n"
      << "RNG stream, so for a given scene and spp they trace the same paths.\n";
}

// False on a malformed or out-of-range value, so the caller can name the flag.
bool parse_int(std::string_view s, int& out, int lo, int hi) {
  int v = 0;
  auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  if (res.ec != std::errc{} || res.ptr != s.data() + s.size()) return false;
  if (v < lo || v > hi) return false;
  out = v;
  return true;
}

} // namespace

int main(int argc, char* argv[]) {
  HeadlessOptions opts;
  const std::vector<std::string> args(argv + 1, argv + argc);

  // Report the flag name rather than a bare index error on a missing value.
  auto need_value = [&](size_t& i, const char* flag) -> const std::string* {
    if (i + 1 >= args.size()) {
      std::cerr << flag << " requires a value\n";
      return nullptr;
    }
    return &args[++i];
  };

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];

    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return EXIT_SUCCESS;

    } else if (a == "--headless") {
      opts.enabled = true;

    } else if (a == "--scene") {
      const std::string* v = need_value(i, "--scene");
      if (!v) return EXIT_FAILURE;
      if (!scene_from_name(*v, opts.scene)) {
        std::cerr << "unknown scene '" << *v << "'; expected one of: " << scene_name_list() << "\n";
        return EXIT_FAILURE;
      }

    } else if (a == "--width") {
      const std::string* v = need_value(i, "--width");
      if (!v || !parse_int(*v, opts.width, 1, 16384)) {
        std::cerr << "--width expects an integer in [1, 16384]\n";
        return EXIT_FAILURE;
      }

    } else if (a == "--spp") {
      const std::string* v = need_value(i, "--spp");
      if (!v || !parse_int(*v, opts.spp, 1, 1000000)) {
        std::cerr << "--spp expects an integer in [1, 1000000]\n";
        return EXIT_FAILURE;
      }

    } else if (a == "--depth") {
      const std::string* v = need_value(i, "--depth");
      if (!v || !parse_int(*v, opts.max_depth, 1, 1024)) {
        std::cerr << "--depth expects an integer in [1, 1024]\n";
        return EXIT_FAILURE;
      }

    } else if (a == "--backend") {
      const std::string* v = need_value(i, "--backend");
      if (!v) return EXIT_FAILURE;
      if (*v == "gpu") opts.backend = HeadlessOptions::Backend::GPU;
      else if (*v == "cpu") opts.backend = HeadlessOptions::Backend::CPU;
      else if (*v == "both") opts.backend = HeadlessOptions::Backend::BOTH;
      else {
        std::cerr << "--backend expects gpu, cpu or both (got '" << *v << "')\n";
        return EXIT_FAILURE;
      }

    } else if (a == "--out") {
      const std::string* v = need_value(i, "--out");
      if (!v) return EXIT_FAILURE;
      opts.out = *v;

    } else {
      std::cerr << "unknown option '" << a << "'\n\n";
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (!opts.enabled && opts.backend != HeadlessOptions::Backend::GPU) {
    std::cerr << "--backend only applies with --headless; the GUI has its own toggle\n";
    return EXIT_FAILURE;
  }

  try {
    VulkanApp app(opts);
    app.run();
  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
