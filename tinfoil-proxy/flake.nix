{
  description = "tinfoil-proxy Go tooling";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        packages.default = pkgs.buildGoModule {
          pname = "tinfoil-proxy";
          version = "0.1.0";
          src = ./.;
          vendorHash = null;
          env.CGO_ENABLED = 0;
          ldflags = [ "-s" "-w" ];
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.go_1_26
            pkgs.gopls
          ];
        };
      });
}
