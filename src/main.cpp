#include "vulkan_app.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  bool headless = false;

  // Simple argument parsing
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--headless") {
      headless = true;
    }
  }

  try {
    VulkanApp app(headless);
    app.run();
  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
