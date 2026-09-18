{
  description = "CUDA + Vulkan raytracer devShell";

  inputs = {
    nixos-config.url = "path:/home/neel/.nixos";
    nixpkgs.follows = "nixos-config/nixpkgs";
  };

  outputs = { self, nixpkgs, nixos-config, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in
    {
      devShells.${system}.default = nixos-config.lib.mkCudaShell {
        inherit pkgs;
        cudaArch = "75";

        extraPackages = with pkgs; [
          cmake
          glfw
          vulkan-headers
          vulkan-loader
          vulkan-validation-layers
          vulkan-tools
          shaderc # glslc
          pkg-config
          clang-tools # clangd
        ];

        extraHook = ''
          # nvcc rejects host compilers it doesn't explicitly support; pin to the
          # compiler nixpkgs pairs with this cudaPackages set instead of the
          # (possibly too-new) default gcc pulled in by cuda/default.nix.
          export CC="${pkgs.cudaPackages.backendStdenv.cc}/bin/cc"
          export CXX="${pkgs.cudaPackages.backendStdenv.cc}/bin/c++"
          export CUDAHOSTCXX="$CXX"

          export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"

          # Stable path for .clangd (which can't embed a nix store hash) to
          # point --cuda-path at.
          ln -sfn "${pkgs.cudaPackages.cudatoolkit}" ./.nix-cuda-sdk
        '';
      };
    };
}
