{
  description = "Rust project";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "nixpkgs/nixos-unstable";
    # Keep dev security tools current without moving the application or Nitro build pin.
    security-tools-nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-26.05-darwin";
    nitro-util = {
      url = "github:monzo/aws-nitro-util/7d755578b0b0b9850c0d7c4738a6c8daf3ff55c0";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, security-tools-nixpkgs, flake-utils, rust-overlay, nitro-util }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ rust-overlay.overlays.default ];
        pkgs = import nixpkgs { inherit system overlays; };
        securityToolsPkgs = import security-tools-nixpkgs { inherit system; };
        rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        nitro = nitro-util.lib.${system};
        kernelUpstream = import ./nix/kernel-upstream.nix;

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
        securityToolInputs = [
          securityToolsPkgs.cargo-audit
          securityToolsPkgs.cargo-deny
          securityToolsPkgs.cargo-machete
        ];
        inputs = commonInputs
          ++ securityToolInputs
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux linuxOnlyInputs
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin darwinOnlyInputs;

        setupPostgresScript = pkgs.writeShellScript "setup-postgres" ''
          case "''${OPENSECRET_DEV_POSTGRES:-1}" in
            0|false|False|FALSE|no|No|NO|skip|Skip|SKIP)
              exit 0
              ;;
          esac

          export PGDATA="''${PGDATA:-$PWD/.pgdata}"
          export PGPORT="''${PGPORT:-5432}"
          export PGSOCKETS="''${PGSOCKETS:-$PGDATA/sockets}"

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
          case "''${OPENSECRET_DEV_ENV:-1}" in
            0|false|False|FALSE|no|No|NO|skip|Skip|SKIP)
              exit 0
              ;;
          esac

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

            export PGPORT="''${PGPORT:-5432}"
            export OPENSECRET_DEV_DATABASE_URL="''${OPENSECRET_DEV_DATABASE_URL:-postgres://opensecret_user:password@localhost:$PGPORT/opensecret}"
            replace_env 'DATABASE_URL=postgres://localhost/opensecret' "DATABASE_URL=$OPENSECRET_DEV_DATABASE_URL"

            # Get a new ENCLAVE_SECRET_MOCK value using openssl
            export enclaveSecret=$(openssl rand -hex 32)
            replace_env 'ENCLAVE_SECRET_MOCK=' "ENCLAVE_SECRET_MOCK=$enclaveSecret"

            # Get a new JWT_SECRET value using openssl
            export jwtSecret=$(openssl rand -base64 32)
            replace_env 'JWT_SECRET=' "JWT_SECRET=$jwtSecret"
          fi
        '';

        # Function to create rootfs with specific APP_MODE
        mkRootfs = { appMode, opensecretPkg ? opensecret }: pkgs.buildEnv {
          name = "opensecret-rootfs-${appMode}";
          paths = [
            opensecretPkg
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

              # Copy opensecret and the remaining Continuum sidecar.
              mkdir -p /app
              install -m 755 ${opensecretPkg}/bin/opensecret /app/
              install -m 755 ${continuum-proxy}/bin/continuum-proxy /app/

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
          ];
          pathsToLink = [ "/bin" "/lib" "/app" "/usr/bin" "/usr/sbin" "/sbin" ];
        };

        # Nixpkgs remains the build framework, but the enclave kernel's stable
        # patch level is pinned directly to kernel.org so security fixes do not
        # wait for a Nixpkgs refresh.
        expectedKernelUrl =
          "https://cdn.kernel.org/pub/linux/kernel/v${pkgs.lib.versions.major kernelUpstream.version}.x/linux-${kernelUpstream.version}.tar.xz";
        kernelSource = pkgs.fetchurl {
          inherit (kernelUpstream) url hash;
        };
        kernelStructuredExtraConfig = with pkgs.lib.kernel; {
          VIRTIO = yes;
          VIRTIO_MMIO = yes;
          VIRTIO_MENU = yes;
          VIRTIO_MMIO_CMDLINE_DEVICES = yes;
          NET = yes;
          VSOCKETS = yes;
          VIRTIO_VSOCKETS = yes;
          HW_RANDOM = yes;
          NSM = yes; # Enable the in-tree NSM driver for KMS operations.
          # Keep the unused AF_ALG AEAD socket adapter disabled. This does not
          # affect OpenSecret's userspace AEAD implementations.
          CRYPTO_USER_API_AEAD = no;
        };
        unverifiedCustomKernel = pkgs.linux_6_12.override {
          argsOverride = {
            inherit (kernelUpstream) version;
            modDirVersion = kernelUpstream.version;
            src = kernelSource;
          };
          structuredExtraConfig = kernelStructuredExtraConfig;
          # Fail if a required option is renamed or becomes unavailable.
          ignoreConfigErrors = false;
        };
        customKernel =
          assert pkgs.lib.assertMsg (
            kernelUpstream.branch == "6.12"
          ) "The enclave kernel must remain on the reviewed 6.12 LTS branch";
          assert pkgs.lib.assertMsg (
            pkgs.lib.versions.majorMinor kernelUpstream.version == kernelUpstream.branch
          ) "The enclave kernel version must match its declared LTS branch";
          assert pkgs.lib.assertMsg (
            kernelUpstream.url == expectedKernelUrl
          ) "The enclave kernel URL must be the version-matched kernel.org stable tarball";
          assert pkgs.lib.assertMsg (
            pkgs.lib.hasPrefix "sha256-" kernelUpstream.hash
          ) "The enclave kernel source must use an immutable SRI sha256 hash";
          assert pkgs.lib.assertMsg (
            unverifiedCustomKernel.version == kernelUpstream.version
          ) "The realized enclave kernel version differs from the direct upstream pin";
          assert pkgs.lib.assertMsg (
            unverifiedCustomKernel.modDirVersion == kernelUpstream.version
          ) "The realized enclave kernel module version differs from the direct upstream pin";
          assert pkgs.lib.assertMsg (
            kernelStructuredExtraConfig.NSM == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.HW_RANDOM == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VIRTIO == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VIRTIO_MMIO == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VIRTIO_MENU == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VIRTIO_MMIO_CMDLINE_DEVICES == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.NET == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VSOCKETS == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VIRTIO_VSOCKETS == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.CRYPTO_USER_API_AEAD == pkgs.lib.kernel.no
          ) "The enclave kernel's required NSM, hwrng, virtio, vsock, networking, or AEAD configuration changed";
          unverifiedCustomKernel;

        kernelSourcePin = pkgs.runCommand "opensecret-kernel-source-pin-${kernelUpstream.version}"
          {
            nativeBuildInputs = [
              pkgs.gnugrep
              pkgs.gnutar
              pkgs.xz
            ];
          }
          ''
            set -euo pipefail

            test -s ${kernelSource}
            tar -xOf ${kernelSource} \
              linux-${kernelUpstream.version}/drivers/misc/nsm.c > nsm.c

            # 6.12.101 must carry both reviewed NSM fixes: malformed userspace
            # pointers return before the mutex is touched, and file operations
            # retain module ownership.
            ioctl_source="$(sed -n \
              '/static long nsm_dev_ioctl/,/static int nsm_device_init_vq/p' \
              nsm.c)"
            printf '%s\n' "$ioctl_source" > nsm_dev_ioctl.c

            efault_line="$(grep -nF 'r = -EFAULT;' nsm_dev_ioctl.c | head -n1 | cut -d: -f1)"
            copy_line="$(grep -nF 'if (copy_from_user(&raw, argp, _IOC_SIZE(cmd)))' nsm_dev_ioctl.c | cut -d: -f1)"
            return_line="$(grep -nF 'return r;' nsm_dev_ioctl.c | head -n1 | cut -d: -f1)"
            mutex_line="$(grep -nF 'mutex_lock(&nsm->lock);' nsm_dev_ioctl.c | cut -d: -f1)"

            test -n "$efault_line"
            test -n "$copy_line"
            test -n "$return_line"
            test -n "$mutex_line"
            test "$efault_line" -lt "$copy_line"
            test "$copy_line" -lt "$return_line"
            test "$return_line" -lt "$mutex_line"

            fops_source="$(sed -n \
              '/static const struct file_operations nsm_dev_fops = {/,/};/p' \
              nsm.c)"
            printf '%s\n' "$fops_source" | grep -Fq '.owner = THIS_MODULE,'
            printf '%s\n' "$fops_source" | grep -Fq '.unlocked_ioctl = nsm_dev_ioctl,'

            {
              echo 'branch=${kernelUpstream.branch}'
              echo 'version=${kernelUpstream.version}'
              echo 'source=${kernelUpstream.url}'
              echo 'hash=${kernelUpstream.hash}'
              echo 'store_path=${kernelSource}'
            } > "$out"
          '';

        kernelSecurityInvariants = pkgs.runCommand
          "opensecret-kernel-security-invariants-${kernelUpstream.version}"
          { nativeBuildInputs = [ pkgs.gnugrep ]; }
          ''
            set -euo pipefail

            config=${customKernel.configfile}
            grep -Fqx 'CONFIG_VIRTIO=y' "$config"
            grep -Fqx 'CONFIG_VIRTIO_MMIO=y' "$config"
            grep -Fqx 'CONFIG_VIRTIO_MENU=y' "$config"
            grep -Fqx 'CONFIG_VIRTIO_MMIO_CMDLINE_DEVICES=y' "$config"
            grep -Fqx 'CONFIG_NET=y' "$config"
            grep -Fqx 'CONFIG_VSOCKETS=y' "$config"
            grep -Fqx 'CONFIG_VIRTIO_VSOCKETS=y' "$config"
            grep -Fqx 'CONFIG_HW_RANDOM=y' "$config"
            grep -Fqx 'CONFIG_NSM=y' "$config"
            grep -Fqx '# CONFIG_CRYPTO_USER_API_AEAD is not set' "$config"

            {
              echo 'branch=${kernelUpstream.branch}'
              echo 'version=${kernelUpstream.version}'
              echo 'source=${kernelUpstream.url}'
              echo 'hash=${kernelUpstream.hash}'
              echo 'cmdline=${enclaveKernelCmdline}'
            } > "$out"
          '';

        entrypointEntropyPreflight = pkgs.runCommand
          "opensecret-entrypoint-entropy-preflight"
          {
            nativeBuildInputs = [
              pkgs.bash
              pkgs.coreutils
              pkgs.gnused
            ];
          }
          ''
            ENTRYPOINT_UNDER_TEST=${./entrypoint.sh} \
              bash ${./tests/entrypoint_entropy_preflight.sh}
            touch "$out"
          '';

        # aws-nitro-util already source-builds this init by default. Keep the
        # same source and build flags explicit while using the current
        # buildGoModule API, so a Nixpkgs refresh cannot revive the deprecated
        # top-level CGO_ENABLED argument.
        nitroInitSrc = builtins.path {
          path = "${nitro-util}/init";
          name = "init";
        };
        nitroInit = pkgs.buildGoModule {
          name = "eif-init";
          src = nitroInitSrc;
          vendorHash = null;
          env.CGO_ENABLED = "0";
          ldflags = [
            "-s"
            "-w"
          ];
        };

        # Preserve the currently deployed Nitro boot behavior explicitly. A
        # trust-policy experiment belongs in its own held draft PR.
        enclaveKernelCmdline =
          "reboot=k panic=30 pci=off nomodules console=ttyS0 random.trust_cpu=on root=/dev/ram0";

        # Function to create EIF with specific APP_MODE
        mkEif = { appMode, opensecretPkg ? opensecret, nameSuffix ? "" }: nitro.buildEif {
          name = "opensecret-eif-${appMode}${nameSuffix}";
          # The kernel image location varies by architecture
          kernel = if arch == "aarch64"
            then "${customKernel}/Image"  # ARM64 uses Image
            else "${customKernel}/bzImage"; # x86_64 uses bzImage
          # EIF metadata must describe the exact configuration used to build
          # this kernel, not the unrelated pre-built Nitro blob configuration.
          kernelConfig = customKernel.configfile;
          cmdline = enclaveKernelCmdline;
          # NSM driver is built into kernel 6.8+, so we don't need the old module
          # Setting to null to skip loading the incompatible old module
          nsmKo = null;
          copyToRoot = mkRootfs { inherit appMode opensecretPkg; };
          entrypoint = "/bin/entrypoint";
          init = "${nitroInit}/bin/init";
        };

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
            outputHashes = {
              "tinfoil-0.1.0" = "sha256-ViCg20dzw61r1k740xQpyJjfBthv6yXHzBAhxH7OC8Y=";
            };
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

        arch = pkgs.stdenv.hostPlatform.uname.processor;
      in
      {
        packages = {
          default = opensecret;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          nitro-init = nitroInit;
          eif-dev = mkEif { appMode = "dev"; };
          eif-prod = mkEif { appMode = "prod"; };
          eif-preview = mkEif { appMode = "preview"; };
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

        checks = {
          entrypoint-entropy-preflight = entrypointEntropyPreflight;
          kernel-source-pin = kernelSourcePin;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          kernel-security-invariants = kernelSecurityInvariants;
        };

        devShell = pkgs.mkShell {
          packages = inputs;
          shellHook = ''
            export PGDATA="''${PGDATA:-$PWD/.pgdata}"
            export PGPORT="''${PGPORT:-5432}"
            export PGSOCKETS="''${PGSOCKETS:-$PGDATA/sockets}"
            export OPENSECRET_DEV_DATABASE_URL="''${OPENSECRET_DEV_DATABASE_URL:-postgres://opensecret_user:password@localhost:$PGPORT/opensecret}"

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
