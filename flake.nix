{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }: let
    system = "aarch64-darwin";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        pkgs.llama-cpp
        pkgs.cmake
        pkgs.ninja
        pkgs.yaml-cpp
        pkgs.pkg-config
        pkgs.nlohmann_json
        pkgs.cli11
        pkgs.curl.dev
      ];
    };
  };
}
