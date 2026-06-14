{
  description = "Rust project";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "nixpkgs/nixos-unstable";
    nitro-util = {
      url = "github:monzo/aws-nitro-util/7d755578b0b0b9850c0d7c4738a6c8daf3ff55c0";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, nitro-util }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ rust-overlay.overlays.default ];
        pkgs = import nixpkgs { inherit system overlays; };
        rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        nitro = nitro-util.lib.${system};

        # Development environment setup
        # Get rust-analyzer matching the channel in rust-toolchain.toml
        rustToolchain = builtins.fromTOML (builtins.readFile ./rust-toolchain.toml);
        rustChannel = rustToolchain.toolchain.channel;
        rustAnalyzer = pkgs.rust-bin.stable."${rustChannel}".rust-analyzer;

        commonInputs = [
          rust
          rustAnalyzer
          pkgs.pkg-config
          pkgs.openssl
          pkgs.zlib
          pkgs.gcc
          pkgs.clang
          pkgs.jq
          pkgs.just
          pkgs.postgresql
          pkgs.diesel-cli
          pkgs.python3
          (pkgs.python3.withPackages (ps: with ps; [
            cryptography
          ]))
          pkgs.go
        ];
        linuxOnlyInputs = [
          pkgs.podman
          pkgs.conmon
          pkgs.slirp4netns
          pkgs.fuse-overlayfs
        ];
        darwinOnlyInputs = [
          pkgs.libiconv
          pkgs.apple-sdk
        ];
        inputs = commonInputs
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux linuxOnlyInputs
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin darwinOnlyInputs;

        setupPostgresScript = pkgs.writeShellScript "setup-postgres" ''
          export PGDATA="$PWD/.pgdata"
          export PGPORT=5432
          export PGSOCKETS="$PWD/.pgdata/sockets"

          # Skip if Postgres is already running
          if ${pkgs.postgresql}/bin/pg_isready -h localhost -p $PGPORT >/dev/null 2>&1; then
            exit 0
          fi

          # Initialize if needed
          if [ ! -f "$PGDATA/PG_VERSION" ]; then
            ${pkgs.postgresql}/bin/initdb -D "$PGDATA"
          fi

          # Ensure socket directory exists
          mkdir -p "$PGSOCKETS"

          # Start Postgres
          ${pkgs.postgresql}/bin/pg_ctl start -D "$PGDATA" -o "-h localhost -p $PGPORT -k $PGSOCKETS" -l "$PGDATA/logfile" -w

          # Wait for it to be ready
          until ${pkgs.postgresql}/bin/pg_isready -h localhost -p $PGPORT >/dev/null 2>&1; do sleep 0.5; done

          # Create user and database if they don't exist
          ${pkgs.postgresql}/bin/psql -h localhost -p $PGPORT -tc "SELECT 1 FROM pg_roles WHERE rolname='opensecret_user'" postgres 2>/dev/null | grep -q 1 || \
            ${pkgs.postgresql}/bin/psql -h localhost -p $PGPORT -c "CREATE USER \"opensecret_user\" WITH PASSWORD 'password';" postgres
          ${pkgs.postgresql}/bin/psql -h localhost -p $PGPORT -tc "SELECT 1 FROM pg_database WHERE datname='opensecret'" postgres 2>/dev/null | grep -q 1 || \
            ${pkgs.postgresql}/bin/psql -h localhost -p $PGPORT -c "CREATE DATABASE \"opensecret\" OWNER \"opensecret_user\";" postgres
        '';

        setupEnvScript = pkgs.writeShellScript "setup-env" ''
          if [ ! -f .env ]; then
            cp .env.sample .env

            replace_env() {
              local pattern="$1"
              local replacement="$2"
              local tmp
              tmp="$(mktemp)"
              sed "s|$pattern|$replacement|g" .env > "$tmp"
              mv "$tmp" .env
            }

            replace_env 'DATABASE_URL=postgres://localhost/opensecret' 'DATABASE_URL=postgres://opensecret_user:password@localhost:5432/opensecret'

            # Get a new ENCLAVE_SECRET_MOCK value using openssl
            export enclaveSecret=$(openssl rand -hex 32)
            replace_env 'ENCLAVE_SECRET_MOCK=' "ENCLAVE_SECRET_MOCK=$enclaveSecret"

            # Get a new JWT_SECRET value using openssl
            export jwtSecret=$(openssl rand -base64 32)
            replace_env 'JWT_SECRET=' "JWT_SECRET=$jwtSecret"
          fi
        '';

        # Function to create rootfs with specific APP_MODE
        mkRootfs = appMode: pkgs.buildEnv {
          name = "opensecret-rootfs-${appMode}";
          paths = [
            opensecret
            (pkgs.writeScriptBin "entrypoint" ''
              #!${pkgs.bash}/bin/bash

              # Set up busybox commands and other tools
              export PATH="/bin:${pkgs.busybox}/bin:${pkgs.python3}/bin:${pkgs.jq}/bin:${pkgs.socat}/bin:${nitro-bins}/bin:$PATH"

              # Create symlinks for busybox commands
              mkdir -p /bin
              ln -sf ${pkgs.busybox}/bin/busybox /bin/date
              ln -sf ${pkgs.busybox}/bin/busybox /bin/ip
              ln -sf ${pkgs.python3}/bin/python3 /bin/python3
              ln -sf ${pkgs.jq}/bin/jq /bin/jq
              ln -sf ${pkgs.socat}/bin/socat /bin/socat
              ln -sf ${pkgs.curl}/bin/curl /bin/curl

              # Set up CA certificates
              mkdir -p /etc/ssl/certs
              ln -sf ${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-bundle.crt
              export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
              export AWS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

              # Copy required libraries and tools
              mkdir -p /lib
              export LD_LIBRARY_PATH="/lib:$LD_LIBRARY_PATH"
              install -m 755 ${nitro-bins}/lib/libnsm.so /lib/

              install -m 755 ${nitro-bins}/bin/kmstool_enclave_cli /bin/

              # Copy required C libraries
              cp -P ${pkgs.glibc}/lib/ld-linux*.so* /lib/
              cp -P ${pkgs.glibc}/lib/libc.so* /lib/
              cp -P ${pkgs.glibc}/lib/libdl.so* /lib/
              cp -P ${pkgs.glibc}/lib/libpthread.so* /lib/
              cp -P ${pkgs.glibc}/lib/librt.so* /lib/
              cp -P ${pkgs.glibc}/lib/libm.so* /lib/

              # Set up Python environment
              export PYTHONPATH="$(find ${pkgs.python3}/lib -name site-packages):$PYTHONPATH"

              # Copy opensecret and continuum-proxy to their locations
              mkdir -p /app
              install -m 755 ${opensecret}/bin/opensecret /app/
              install -m 755 ${continuum-proxy}/bin/continuum-proxy /app/
              install -m 755 ${tinfoil-proxy}/bin/tinfoil-proxy /app/

              ${builtins.readFile ./entrypoint.sh}
            '')
            (pkgs.writeTextFile {
              name = "app-mode";
              text = builtins.trace "Creating APP_MODE file with value: ${appMode}" appMode;
              destination = "/app/APP_MODE";
            })
            (pkgs.writeTextFile {
              name = "traffic_forwarder";
              text = builtins.readFile ./nitro-toolkit/traffic_forwarder.py;
              destination = "/app/traffic_forwarder.py";
            })
            (pkgs.writeTextFile {
              name = "vsock_helper";
              text = builtins.readFile ./nitro-toolkit/vsock_helper.py;
              destination = "/app/vsock_helper.py";
            })
            pkgs.bash
            pkgs.busybox
            pkgs.openssl
            pkgs.postgresql
            pkgs.socat
            pkgs.python3
            pkgs.jq
            pkgs.iproute2
            pkgs.coreutils
            pkgs.cacert
            pkgs.curl
            nitro-bins
            continuum-proxy
            tinfoil-proxy
          ];
          pathsToLink = [ "/bin" "/lib" "/app" "/usr/bin" "/usr/sbin" "/sbin" ];
        };

        # Build custom kernel - use kernel 6.12 which has NSM driver (merged in 6.8)
        customKernel = pkgs.linuxPackages_6_12.kernel.override {
          structuredExtraConfig = with pkgs.lib.kernel; {
            VIRTIO_MMIO = yes;
            VIRTIO_MENU = yes;
            VIRTIO_MMIO_CMDLINE_DEVICES = yes;
            NET = yes;
            VSOCKETS = yes;
            VIRTIO_VSOCKETS = yes;
            NSM = yes;  # Enable NSM driver for KMS operations (merged in 6.8+)
            # Disable algif_aead, the AF_ALG AEAD interface abused by CVE-2026-31431 (Copy Fail).
            CRYPTO_USER_API_AEAD = no;
          };
          # Ensure we catch invalid or renamed config flags at build time
          ignoreConfigErrors = false;
        };

        # Function to create EIF with specific APP_MODE
        mkEif = appMode: nitro.buildEif {
          name = "opensecret-eif-${appMode}";
          # The kernel image location varies by architecture
          kernel = if arch == "aarch64"
            then "${customKernel}/Image"  # ARM64 uses Image
            else "${customKernel}/bzImage"; # x86_64 uses bzImage
          # Use the blob config since extracting from custom kernel is complex
          # The important thing is the kernel itself, not the config file
          kernelConfig = nitro.blobs.${arch}.kernelConfig;
          # NSM driver is built into kernel 6.8+, so we don't need the old module
          # Setting to null to skip loading the incompatible old module
          nsmKo = null;
          copyToRoot = mkRootfs appMode;
          entrypoint = "/bin/entrypoint";
        };

        # Build the main Rust package
        opensecret = pkgs.rustPlatform.buildRustPackage {
          pname = "opensecret";
          version = "0.1.0";
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = path: type:
              let
                baseName = baseNameOf path;
                parts = pkgs.lib.splitString "/" path;
              in
                # Explicitly exclude .env files
                (baseName != ".env" && baseName != ".env.sample") &&
                (
                  (builtins.elem "src" parts) ||
                  (type == "regular" && (
                    baseName == "Cargo.toml" ||
                    baseName == "Cargo.lock" ||
                    baseName == "rust-toolchain.toml"
                  ))
                );
          };
          cargoLock = {
            lockFile = ./Cargo.lock;
          };
          nativeBuildInputs = [
            pkgs.pkg-config
            rustAnalyzer
            pkgs.gcc
            pkgs.clang
          ];
          buildInputs = [
            pkgs.openssl
            pkgs.zlib
            pkgs.postgresql
            pkgs.diesel-cli
          ];
          LIBPQ_LIB_DIR = "${pkgs.postgresql.lib}/lib";
        };

        # Use pre-built NSM library and KMS tools from nitro-bins directory
        nitro-bins = pkgs.stdenv.mkDerivation {
          name = "nitro-bins";
          version = "1.0";
          src = ./nitro-bins;
          dontUnpack = true;
          installPhase = ''
            mkdir -p $out/{lib,bin}
            # Use install to copy files and set permissions
            install -m 755 $src/libnsm.so $out/lib/
            install -m 755 $src/kmstool_enclave_cli $out/bin/
          '';
        };

        nitroBinsBaseImage =
          "public.ecr.aws/amazonlinux/amazonlinux@sha256:89f64859f7faa37ae01fcaab1205a3ae3cfff3f1b98fbb8f6cf489cc9d098508";
        nitroBinsHashes = {
          libnsm = "032f54092d362a479dd69076a68e1344d887c14c085ff0d94065db6b19780644";
          kmstool = "6b151442e024456e52f65e5369a3bb647093618ac516f66e06854f37ec336ade";
        };
        mkNitroBinsApp = { name, writeBins }: pkgs.writeShellApplication {
          inherit name;
          runtimeInputs = [
            pkgs.coreutils
            pkgs.gawk
            pkgs.git
            pkgs.podman
          ];
          text = ''
            set -euo pipefail

            machine="$(uname -m)"
            if [[ "$machine" != "aarch64" && "$machine" != "arm64" ]]; then
              echo "nitro-bins are ARM aarch64 artifacts; run this target on aarch64-linux" >&2
              exit 1
            fi

            repo_root="$(git rev-parse --show-toplevel)"
            work="$(mktemp -d)"
            container_id=""
            cleanup() {
              if [[ -n "$container_id" ]]; then
                podman rm "$container_id" >/dev/null 2>&1 || true
              fi
              rm -rf "$work"
            }
            trap cleanup EXIT

            image_name="''${NITRO_BINS_IMAGE_NAME:-opensecret-nitro-bins-repro}"
            base_image="${nitroBinsBaseImage}"
            expected_lib="${nitroBinsHashes.libnsm}"
            expected_kms="${nitroBinsHashes.kmstool}"

            cp "$repo_root/nitro-toolkit/enclave-base-image/Dockerfile" "$work/Containerfile.in"
            cp "$repo_root/nix/nitro-bins/nsm-api-v0.4.0.Cargo.lock" "$work/Cargo.lock"

            awk -v base_image="$base_image" '
              $0 == "ARG BASE_IMAGE=public.ecr.aws/amazonlinux/amazonlinux:minimal" {
                print "ARG BASE_IMAGE=" base_image
                next
              }
              $0 == "RUN git clone --depth 1 -b v0.4.0 https://github.com/aws/aws-nitro-enclaves-nsm-api.git" {
                print
                print "COPY Cargo.lock /tmp/crt-builder/aws-nitro-enclaves-nsm-api/Cargo.lock"
                next
              }
              $0 == "RUN source $HOME/.cargo/env && cd aws-nitro-enclaves-nsm-api && cargo build --release --jobs $(nproc) -p nsm-lib" {
                print "RUN source $HOME/.cargo/env && cd aws-nitro-enclaves-nsm-api && cargo build --release --locked --jobs $(nproc) -p nsm-lib"
                next
              }
              { print }
            ' "$work/Containerfile.in" > "$work/Containerfile"

            echo "Building nitro-bins image from $base_image"
            podman build --pull=always --no-cache -t "$image_name" -f "$work/Containerfile" "$work"

            podman run --rm "$image_name" sha256sum /app/libnsm.so /app/kmstool_enclave_cli > "$work/hashes"
            cat "$work/hashes"

            actual_lib="$(awk '$2 == "/app/libnsm.so" { print $1 }' "$work/hashes")"
            actual_kms="$(awk '$2 == "/app/kmstool_enclave_cli" { print $1 }' "$work/hashes")"

            if [[ "$actual_lib" != "$expected_lib" || "$actual_kms" != "$expected_kms" ]]; then
              echo "nitro-bins did not match expected hashes" >&2
              echo "expected libnsm.so:           $expected_lib" >&2
              echo "actual   libnsm.so:           $actual_lib" >&2
              echo "expected kmstool_enclave_cli: $expected_kms" >&2
              echo "actual   kmstool_enclave_cli: $actual_kms" >&2
              exit 1
            fi

            echo "nitro-bins match expected hashes"

            ${pkgs.lib.optionalString writeBins ''
              mkdir -p "$repo_root/nitro-bins"
              container_id="$(podman create "$image_name" sh)"
              podman cp "$container_id:/app/libnsm.so" "$work/libnsm.so"
              podman cp "$container_id:/app/kmstool_enclave_cli" "$work/kmstool_enclave_cli"
              install -m 755 "$work/libnsm.so" "$repo_root/nitro-bins/libnsm.so"
              install -m 755 "$work/kmstool_enclave_cli" "$repo_root/nitro-bins/kmstool_enclave_cli"

              written_lib="$(sha256sum "$repo_root/nitro-bins/libnsm.so" | awk '{ print $1 }')"
              written_kms="$(sha256sum "$repo_root/nitro-bins/kmstool_enclave_cli" | awk '{ print $1 }')"
              if [[ "$written_lib" != "$expected_lib" || "$written_kms" != "$expected_kms" ]]; then
                echo "written nitro-bins did not match expected hashes" >&2
                exit 1
              fi

              echo "wrote verified nitro-bins to $repo_root/nitro-bins"
            ''}

            ${pkgs.lib.optionalString (!writeBins) ''
              if [[ -f "$repo_root/nitro-bins/libnsm.so" && -f "$repo_root/nitro-bins/kmstool_enclave_cli" ]]; then
                checked_lib="$(sha256sum "$repo_root/nitro-bins/libnsm.so" | awk '{ print $1 }')"
                checked_kms="$(sha256sum "$repo_root/nitro-bins/kmstool_enclave_cli" | awk '{ print $1 }')"
                if [[ "$checked_lib" != "$actual_lib" || "$checked_kms" != "$actual_kms" ]]; then
                  echo "checked-in nitro-bins do not match reproduced binaries" >&2
                  echo "checked-in libnsm.so:           $checked_lib" >&2
                  echo "reproduced libnsm.so:           $actual_lib" >&2
                  echo "checked-in kmstool_enclave_cli: $checked_kms" >&2
                  echo "reproduced kmstool_enclave_cli: $actual_kms" >&2
                  exit 1
                fi
                echo "nitro-bins match checked-in artifacts"
              else
                echo "checked-in nitro-bins are missing; run write-nitro-bins to regenerate them"
              fi
            ''}
          '';
        };
        reproduceNitroBins = mkNitroBinsApp {
          name = "reproduce-nitro-bins";
          writeBins = false;
        };
        writeNitroBins = mkNitroBinsApp {
          name = "write-nitro-bins";
          writeBins = true;
        };

        # Copy continuum-proxy from local filesystem
        continuum-proxy = pkgs.runCommand "continuum-proxy" {} ''
          mkdir -p $out/bin
          cp ${./continuum-proxy} $out/bin/continuum-proxy
          chmod +x $out/bin/continuum-proxy
        '';

        # Copy tinfoil-proxy from local filesystem
        tinfoil-proxy = pkgs.runCommand "tinfoil-proxy" {} ''
          mkdir -p $out/bin
          cp ${./tinfoil-proxy/dist/tinfoil-proxy} $out/bin/tinfoil-proxy
          chmod +x $out/bin/tinfoil-proxy
        '';

        arch = pkgs.stdenv.hostPlatform.uname.processor;
      in
      {
        packages = {
          default = opensecret;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          eif-dev = mkEif "dev";
          eif-prod = mkEif "prod";
          eif-preview = mkEif "preview";
        };

        apps = pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          reproduce-nitro-bins = {
            type = "app";
            program = "${reproduceNitroBins}/bin/reproduce-nitro-bins";
          };
          write-nitro-bins = {
            type = "app";
            program = "${writeNitroBins}/bin/write-nitro-bins";
          };
        };

        devShell = pkgs.mkShell {
          packages = inputs;
          shellHook = ''
            export LIBCLANG_PATH=${pkgs.libclang.lib}/lib/
            export LD_LIBRARY_PATH=${pkgs.openssl}/lib:$LD_LIBRARY_PATH
            export CC_wasm32_unknown_unknown=${pkgs.llvmPackages_14.clang-unwrapped}/bin/clang-14
            export CFLAGS_wasm32_unknown_unknown="-I ${pkgs.llvmPackages_14.libclang.lib}/lib/clang/14.0.6/include/"
            export PKG_CONFIG_PATH=${pkgs.openssl.dev}/lib/pkgconfig

            ${pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
              export CC=clang
              export CXX=clang++
            ''}

            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              alias docker='podman'
              echo "Using 'podman' as an alias for 'docker'"
              echo "You can now use 'docker' commands, which will be executed by podman"

              # Podman configuration
              export CONTAINERS_CONF=$HOME/.config/containers/containers.conf
              export CONTAINERS_POLICY=$HOME/.config/containers/policy.json
              mkdir -p $HOME/.config/containers
              echo '{"default":[{"type":"insecureAcceptAnything"}]}' > $CONTAINERS_POLICY

              # Create a basic containers.conf if it doesn't exist
              if [ ! -f $CONTAINERS_CONF ]; then
                echo "[engine]
              cgroup_manager = \"cgroupfs\"
              events_logger = \"file\"
              runtime = \"crun\"

              [storage]
              driver = \"vfs\"" > $CONTAINERS_CONF
              fi

              # Ensure correct permissions
              chmod 600 $CONTAINERS_POLICY $CONTAINERS_CONF
            ''}

            ${setupPostgresScript}
            ${setupEnvScript}
          '';
        };
      }
    );
}
