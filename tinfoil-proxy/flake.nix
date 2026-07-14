{
  description = "tinfoil-proxy Go tooling";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        go_1_26_5 = pkgs.go_1_26.overrideAttrs (_: {
          version = "1.26.5";
          src = pkgs.fetchurl {
            url = "https://go.dev/dl/go1.26.5.src.tar.gz";
            hash = "sha256-SVvkvIcXasVnOS5bQRar2YRm0z17SdQedkzMaXay3EI=";
          };
        });
        buildGoModule = pkgs.buildGoModule.override { go = go_1_26_5; };
      in
      {
        packages.default = buildGoModule {
          pname = "tinfoil-proxy";
          version = "0.1.0";
          src = ./.;
          vendorHash = "sha256-8RBh9lSF4dGlr3RbpxjdF8z63Sn2/Rknqmr5BtdlONY=";
          env.CGO_ENABLED = 0;
          ldflags = [ "-s" "-w" ];
        };

        devShells.default = pkgs.mkShell {
          packages = [
            go_1_26_5
            pkgs.gopls
            pkgs.govulncheck
          ];
        };
      });
}
