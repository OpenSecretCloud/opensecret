{
  description = "Rust project";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay/b6916ba032e02122d6ed3064f40cabe937363d43";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # The NSM helper has a newer MSRV than the application. Keep its toolchain
    # independently pinned so helper maintenance cannot silently move the
    # application's compiler or Cargo.lock.
    nitro-rust-overlay = {
      url = "github:oxalica/rust-overlay/b6916ba032e02122d6ed3064f40cabe937363d43";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:NixOS/nixpkgs/531670d871c0e29724a02f3cbcac170adc65b58c";
    # Keep dev security tools current without moving the application or Nitro build pin.
    security-tools-nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-26.05-darwin";
    nitro-util = {
      url = "github:monzo/aws-nitro-util/7d755578b0b0b9850c0d7c4738a6c8daf3ff55c0";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      security-tools-nixpkgs,
      flake-utils,
      rust-overlay,
      nitro-rust-overlay,
      nitro-util,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        securityUpstreams = import ./nix/security-upstreams.nix;
        nixpkgsUpstream = securityUpstreams.nixpkgs;
        nixRuntimeUpstream = securityUpstreams.nixRuntime;
        bashUpstream = securityUpstreams.bash;
        elfutilsUpstream = securityUpstreams.elfutils;
        findutilsUpstream = securityUpstreams.findutils;
        glibcUpstream = securityUpstreams.glibc;
        iproute2Upstream = securityUpstreams.iproute2;
        opensslUpstream = securityUpstreams.openssl;
        appRustUpstream = securityUpstreams.appRust;
        nitroRustUpstream = securityUpstreams.nitroRust;
        nitroUtilUpstream = securityUpstreams.nitroUtil;
        continuumProxyUpstream = securityUpstreams.continuumProxy;
        expectedOpenSslBranch = "3.5";
        expectedOpenSslUrl = "https://github.com/openssl/openssl/releases/download/openssl-${opensslUpstream.version}/openssl-${opensslUpstream.version}.tar.gz";
        # Fetch OpenSSL from an unoverlaid package fixed point. Using final (or
        # prev inside this overlay) can recurse through curl -> final OpenSSL.
        opensslSource = nixpkgs.legacyPackages.${system}.fetchurl {
          inherit (opensslUpstream) url hash;
        };
        mkPinnedOpenSsl =
          prev:
          let
            certificateBundle =
              if prev.stdenv.hostPlatform.isDarwin then
                "/nix/var/nix/profiles/default/etc/ssl/certs/ca-bundle.crt"
              else
                "/etc/ssl/certs/ca-certificates.crt";
            unverifiedOpenSsl = prev.openssl_3_5.overrideAttrs (oldAttrs: {
              version = opensslUpstream.version;
              src = opensslSource;
              # Preserve Nixpkgs' CA bundle behavior. Its 3.5.1 patch no longer
              # applies mechanically after an upstream formatting change, so
              # keep the invariant with a fail-closed source substitution.
              patches = prev.lib.filter (
                patch: !(prev.lib.hasInfix "use-etc-ssl-certs" (builtins.baseNameOf (toString patch)))
              ) (oldAttrs.patches or [ ]);
              postPatch = (oldAttrs.postPatch or "") + ''
                substituteInPlace include/internal/common.h \
                  --replace-fail '#define X509_CERT_FILE OPENSSLDIR "/cert.pem"' \
                                 '#define X509_CERT_FILE "${certificateBundle}"'
              '';
            });
          in
          assert prev.lib.assertMsg (
            opensslUpstream.branch == expectedOpenSslBranch
          ) "OpenSSL must remain on the reviewed ${expectedOpenSslBranch} compatibility branch";
          assert prev.lib.assertMsg (
            prev.lib.versions.majorMinor opensslUpstream.version == opensslUpstream.branch
          ) "The OpenSSL version must match its declared compatibility branch";
          assert prev.lib.assertMsg (
            opensslUpstream.url == expectedOpenSslUrl
          ) "The OpenSSL URL must be the version-matched official upstream release tarball";
          assert prev.lib.assertMsg (prev.lib.hasPrefix "sha256-" opensslUpstream.hash)
            "The OpenSSL source must use an immutable SRI sha256 hash";
          assert prev.lib.assertMsg (
            unverifiedOpenSsl.version == opensslUpstream.version
          ) "The realized OpenSSL version differs from the direct upstream pin";
          unverifiedOpenSsl;
        opensslOverlay =
          _final: prev:
          let
            pinnedOpenSsl = mkPinnedOpenSsl prev;
          in
          {
            openssl = pinnedOpenSsl;
            openssl_3_5 = pinnedOpenSsl;
          };
        glibcBootstrapOverlay = import ./nix/glibc-2.44/overlay.nix {
          actualNixpkgsRev = nixpkgs.rev;
          upstream = glibcUpstream;
        };
        linuxSecurityUpstreamOverlay =
          _final: prev:
          let
            isFinalLinuxGlibcStage =
              prev.stdenv.hostPlatform.isLinux
              && prev.stdenv.name == "stdenv-linux"
              && prev.stdenv.cc.libc.version == glibcUpstream.version;
          in
          prev.lib.optionalAttrs isFinalLinuxGlibcStage (
            let
              expectedBaseElfutilsVersion = "0.194";
              expectedElfutilsPatchFragments = [
                "debug-info-from-env.patch"
                "fix-aarch64_fregs.patch"
                "musl-asm-ptrace-h.patch"
                "musl-macros.patch"
                "musl-strndupa.patch"
                "fix-aarch64_attributes.patch"
                "test-run-sysroot-reliability.patch"
              ];
              patchName = patch: if builtins.isPath patch then builtins.baseNameOf patch else patch.name or "";
              hasPatchFragment = fragment: patch: prev.lib.hasInfix fragment (patchName patch);
              baseElfutils = prev.elfutils;
              pinnedElfutils = baseElfutils.overrideAttrs (
                oldAttrs:
                let
                  patchSetMatches =
                    builtins.length oldAttrs.patches == builtins.length expectedElfutilsPatchFragments
                    && builtins.all (matches: matches) (
                      prev.lib.imap0 (
                        index: fragment: hasPatchFragment fragment (builtins.elemAt oldAttrs.patches index)
                      ) expectedElfutilsPatchFragments
                    );
                in
                assert prev.lib.assertMsg (
                  oldAttrs.version == expectedBaseElfutilsVersion
                ) "The Nixpkgs elfutils base changed; review the direct upstream composition";
                assert prev.lib.assertMsg patchSetMatches
                  "The Nixpkgs elfutils patch set changed; review the direct upstream composition";
                assert prev.lib.assertMsg (
                  (oldAttrs.env.NIX_CFLAGS_COMPILE or "") == ""
                ) "The Nixpkgs elfutils compiler flags changed; review the direct upstream composition";
                {
                  inherit (elfutilsUpstream) version;
                  src = prev.fetchurl {
                    inherit (elfutilsUpstream) url hash;
                  };
                  # 0.195 contains the C23 const-correctness repair. This
                  # overlay targets glibc on aarch64/x86_64, so retain the two
                  # applicable Nixpkgs patches, drop musl-only/upstreamed ones,
                  # and include the current post-release i386 backend fix.
                  patches = prev.lib.take 2 oldAttrs.patches ++ [
                    (prev.fetchpatch {
                      name = "fix-i386_tlsdesc_relocation.patch";
                      inherit (elfutilsUpstream.patches.i386Tlsdesc) url hash;
                    })
                  ];
                  patchFlags = [
                    "-p1"
                    "--fuzz=0"
                    "--batch"
                    "--forward"
                  ];
                }
              );
              expectedKrb5CFlags =
                "-std=gnu17" + prev.lib.optionalString prev.stdenv.hostPlatform.isStatic " -fcommon";
              compatibleKrb5 = prev.krb5.overrideAttrs (
                oldAttrs:
                assert prev.lib.assertMsg (
                  oldAttrs.version == nixRuntimeUpstream.krb5
                ) "The Nixpkgs krb5 version changed; review the glibc 2.44 compatibility override";
                assert prev.lib.assertMsg (
                  (oldAttrs.env.NIX_CFLAGS_COMPILE or "") == expectedKrb5CFlags
                ) "The Nixpkgs krb5 compiler flags changed; review the glibc 2.44 compatibility override";
                {
                  env = (oldAttrs.env or { }) // {
                    NIX_CFLAGS_COMPILE =
                      "-std=gnu17 -Wno-error=discarded-qualifiers"
                      + prev.lib.optionalString prev.stdenv.hostPlatform.isStatic " -fcommon";
                  };
                }
              );
            in
            assert prev.lib.assertMsg (
              elfutilsUpstream.version == nixRuntimeUpstream.elfutils
            ) "The direct elfutils source pin differs from the selected runtime version";
            {
              elfutils = pinnedElfutils;
              # Keep both public names identical so fetchers, kernel build
              # tools, and PostgreSQL cannot select an unpatched variant.
              krb5 = compatibleKrb5;
              libkrb5 = compatibleKrb5;
            }
          );
        # Rebuild the complete Linux/EIF userspace against the direct OpenSSL
        # and glibc pins. glibc is replaced during bootstrap stage 2 so the
        # stdenv, compiler, application, and rootfs cannot form a mixed ABI.
        # On Darwin, use the pinned OpenSSL for the backend and smoke tests
        # without needlessly rebuilding the macOS bootstrap toolchain.
        overlays = [
          rust-overlay.overlays.default
        ]
        ++
          nixpkgs.lib.optionals
            (builtins.elem system [
              "aarch64-linux"
              "x86_64-linux"
            ])
            [
              glibcBootstrapOverlay
              opensslOverlay
              linuxSecurityUpstreamOverlay
            ];
        pkgs = import nixpkgs { inherit system overlays; };
        pinnedOpenSsl =
          if pkgs.stdenv.isLinux then pkgs.openssl else mkPinnedOpenSsl nixpkgs.legacyPackages.${system};
        runtimeElfutils = pkgs.elfutils;
        runtimeKrb5 = pkgs.libkrb5;
        expectedRuntimeKrb5CFlags =
          "-std=gnu17 -Wno-error=discarded-qualifiers"
          + pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isStatic " -fcommon";
        runtimeElfutilsPatchName =
          patch: if builtins.isPath patch then builtins.baseNameOf patch else patch.name or "";
        runtimePostgresql = pkgs.postgresql_17.override {
          openssl = pinnedOpenSsl;
          libkrb5 = runtimeKrb5;
        };
        runtimePython = pkgs.python313;
        runtimeGo = pkgs.go_1_26;
        buildGo126Module = pkgs.buildGoModule.override { go = runtimeGo; };
        runtimeBash = pkgs.bash.overrideAttrs (
          oldAttrs:
          assert pkgs.lib.assertMsg (
            oldAttrs.patch_suffix == "p9" && builtins.length oldAttrs.patches == 10
          ) "The Nixpkgs Bash base changed; review the direct upstream patch composition";
          {
            inherit (bashUpstream) version;
            src = pkgs.fetchurl {
              inherit (bashUpstream) url hash;
            };
            # Preserve Nixpkgs patches 001-009 and its final PGRP_PIPE patch,
            # inserting the independently pinned GNU patches in release order.
            patches =
              (pkgs.lib.init oldAttrs.patches)
              ++ map (
                patch:
                pkgs.fetchurl {
                  url = "https://ftp.gnu.org/gnu/bash/bash-${bashUpstream.baseVersion}-patches/bash53-${patch.number}";
                  inherit (patch) hash;
                }
              ) bashUpstream.patches
              ++ [ (pkgs.lib.last oldAttrs.patches) ];
          }
        );
        runtimeFindutils = pkgs.findutils.overrideAttrs (
          oldAttrs:
          assert pkgs.lib.assertMsg (
            builtins.length oldAttrs.patches == 2
            && builtins.any (
              patch: pkgs.lib.hasInfix "no-install-statedir.patch" (toString patch)
            ) oldAttrs.patches
            && builtins.any (patch: pkgs.lib.hasInfix "gnulib-float-h-tests" (toString patch)) oldAttrs.patches
          ) "The Nixpkgs findutils patch set changed; review the direct upstream composition";
          {
            inherit (findutilsUpstream) version;
            src = pkgs.fetchurl {
              inherit (findutilsUpstream) url hash;
            };
            # 4.11 already contains the gnulib C23/PowerPC changes carried by
            # Nixpkgs for 4.10. Preserve only the NixOS state-directory patch.
            patches = pkgs.lib.filter (
              patch: !(pkgs.lib.hasInfix "gnulib-float-h-tests" (toString patch))
            ) oldAttrs.patches;
          }
        );
        runtimeIproute2 = pkgs.iproute2.overrideAttrs (_oldAttrs: {
          inherit (iproute2Upstream) version;
          src = pkgs.fetchurl {
            inherit (iproute2Upstream) url hash;
          };
        });
        iproute2BinCompat = pkgs.runCommand "iproute2-bin-compat" { } ''
          mkdir -p "$out/bin"
          ln -s ${runtimeIproute2}/sbin/ip "$out/bin/ip"
        '';
        nitroRustPkgs = import nixpkgs {
          inherit system;
          overlays = [ nitro-rust-overlay.overlays.default ];
        };
        nitroRustToolchainVersion =
          assert pkgs.lib.assertMsg (
            nitro-rust-overlay.rev == nitroRustUpstream.overlayRev
            && nitro-rust-overlay.narHash == nitroRustUpstream.overlayHash
          ) "The Nitro helper rust-overlay input differs from the reviewed immutable pin";
          nitroRustUpstream.version;
        nitroRustToolchain = nitroRustPkgs.rust-bin.stable."${nitroRustToolchainVersion}".minimal;
        securityToolsPkgs = import security-tools-nixpkgs { inherit system; };

        # Keep every application build surface on the same explicitly reviewed
        # compiler instead of inheriting pkgs.rustPlatform from the snapshot.
        rustToolchain = builtins.fromTOML (builtins.readFile ./rust-toolchain.toml);
        rustChannel = rustToolchain.toolchain.channel;
        rust =
          assert pkgs.lib.assertMsg (
            rust-overlay.rev == appRustUpstream.overlayRev
            && rust-overlay.narHash == appRustUpstream.overlayHash
          ) "The application rust-overlay input differs from the reviewed immutable pin";
          assert pkgs.lib.assertMsg (
            rustChannel == appRustUpstream.version
          ) "rust-toolchain.toml differs from the reviewed application Rust pin";
          pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        appRustPlatform = pkgs.makeRustPlatform {
          cargo = rust;
          rustc = rust;
        };
        runtimeDieselCli = pkgs.callPackage (nixpkgs.outPath + "/pkgs/by-name/di/diesel-cli/package.nix") {
          libpq = runtimePostgresql.lib;
          mysqlSupport = false;
          openssl = pinnedOpenSsl;
          rustPlatform = appRustPlatform;
          sqliteSupport = false;
        };
        rustAnalyzer = pkgs.rust-bin.stable."${rustChannel}".rust-analyzer;
        nitro = nitro-util.lib.${system};

        commonInputs = [
          rust
          rustAnalyzer
          pkgs.pkg-config
          pinnedOpenSsl
          pkgs.zlib
          pkgs.gcc
          pkgs.clang
          pkgs.jq
          pkgs.just
          runtimePostgresql
          runtimeDieselCli
          runtimePython
          (runtimePython.withPackages (
            ps: with ps; [
              cryptography
            ]
          ))
          runtimeGo
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
        inputs =
          commonInputs
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

          # Reuse only this workspace's exact pinned server. A listener from a
          # different checkout or an older closure would make local smoke tests
          # look current while exercising the wrong OpenSSL/PostgreSQL build.
          if ${runtimePostgresql}/bin/pg_isready -h localhost -p $PGPORT >/dev/null 2>&1; then
            if ! ${runtimePostgresql}/bin/pg_ctl status -D "$PGDATA" >/dev/null 2>&1 \
              || [ ! -f "$PGDATA/postmaster.opts" ]; then
              echo "PostgreSQL port $PGPORT is occupied by a server outside $PGDATA" >&2
              echo "Choose another PGPORT or stop that server explicitly; it will not be replaced automatically." >&2
              exit 1
            fi

            read -r running_postgres _ < "$PGDATA/postmaster.opts"
            if [ "$running_postgres" != "${runtimePostgresql}/bin/postgres" ]; then
              echo "PostgreSQL in $PGDATA uses $running_postgres" >&2
              echo "Expected the pinned server ${runtimePostgresql}/bin/postgres; stop or migrate it explicitly." >&2
              exit 1
            fi
            exit 0
          fi

          # Initialize if needed
          if [ ! -f "$PGDATA/PG_VERSION" ]; then
            ${runtimePostgresql}/bin/initdb -D "$PGDATA"
          fi

          # Ensure socket directory exists
          mkdir -p "$PGSOCKETS"

          # Start Postgres
          ${runtimePostgresql}/bin/pg_ctl start -D "$PGDATA" -o "-h localhost -p $PGPORT -k $PGSOCKETS" -l "$PGDATA/logfile" -w

          # Wait for it to be ready
          until ${runtimePostgresql}/bin/pg_isready -h localhost -p $PGPORT >/dev/null 2>&1; do sleep 0.5; done

          # Create user and database if they don't exist
          ${runtimePostgresql}/bin/psql -h localhost -p $PGPORT -tc "SELECT 1 FROM pg_roles WHERE rolname='opensecret_user'" postgres 2>/dev/null | grep -q 1 || \
            ${runtimePostgresql}/bin/psql -h localhost -p $PGPORT -c "CREATE USER \"opensecret_user\" WITH PASSWORD 'password';" postgres
          ${runtimePostgresql}/bin/psql -h localhost -p $PGPORT -tc "SELECT 1 FROM pg_database WHERE datname='opensecret'" postgres 2>/dev/null | grep -q 1 || \
            ${runtimePostgresql}/bin/psql -h localhost -p $PGPORT -c "CREATE DATABASE \"opensecret\" OWNER \"opensecret_user\";" postgres
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

        vsockHelper = pkgs.runCommand "vsock-helper-hardened" { nativeBuildInputs = [ pkgs.patch ]; } ''
          install -m 0644 ${./nitro-toolkit/vsock_helper.py} vsock_helper.py
          patch -p1 --fuzz=0 --batch --forward < ${./nix/vsock-helper-hardening.patch}
          ${runtimePython}/bin/python3 -m py_compile vsock_helper.py
          VSOCK_HELPER_UNDER_TEST="$PWD/vsock_helper.py" \
            ${runtimePython}/bin/python3 ${./tests/vsock_helper_hardening.py}
          grep -F 'MAX_RESPONSE_BYTES = 1024 * 1024' vsock_helper.py >/dev/null
          mkdir -p "$out/app"
          install -m 0444 vsock_helper.py "$out/app/vsock_helper.py"
        '';

        # Function to create rootfs with specific APP_MODE
        mkRootfs =
          {
            appMode,
            opensecretPkg ? opensecret,
          }:
          pkgs.buildEnv {
            name = "opensecret-rootfs-${appMode}";
            paths = [
              opensecretPkg
              nitro-bins
              (pkgs.writeScriptBin "entrypoint" ''
                #!${runtimeBash}/bin/bash

                # Use only the explicitly composed enclave command closure.
                export PATH="/sbin:/usr/sbin:/bin:/usr/bin:${runtimePython}/bin:${pkgs.jq}/bin:${pkgs.socat}/bin:${nitro-bins}/bin:$PATH"

                # Preserve the historical absolute command locations.
                mkdir -p /bin
                ln -sf ${runtimePython}/bin/python3 /bin/python3
                ln -sf ${pkgs.jq}/bin/jq /bin/jq
                ln -sf ${pkgs.socat}/bin/socat /bin/socat

                # Set up CA certificates
                mkdir -p /etc/ssl/certs
                ln -sf ${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-bundle.crt
                export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
                export AWS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

                # Copy required libraries and tools
                mkdir -p /lib
                export LD_LIBRARY_PATH="/lib:$LD_LIBRARY_PATH"
                # kmstool links to the versioned NSM ABI. Keep the complete
                # SONAME chain and Rust unwind runtime together in /lib.
                cp -P ${nitro-bins}/lib/libnsm.so* /lib/
                install -m 755 ${nitro-bins}/lib/libgcc_s.so.1 /lib/

                install -m 755 ${nitro-bins}/bin/kmstool_enclave_cli /bin/

                # Copy required C libraries
                cp -P ${pkgs.glibc}/lib/ld-linux*.so* /lib/
                cp -P ${pkgs.glibc}/lib/libc.so* /lib/
                cp -P ${pkgs.glibc}/lib/libdl.so* /lib/
                cp -P ${pkgs.glibc}/lib/libpthread.so* /lib/
                cp -P ${pkgs.glibc}/lib/librt.so* /lib/
                cp -P ${pkgs.glibc}/lib/libm.so* /lib/

                # Set up Python environment
                export PYTHONPATH="$(find ${runtimePython}/lib -name site-packages):$PYTHONPATH"

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
              vsockHelper
              runtimeBash
              pkgs.openssl
              pkgs.socat
              runtimePython
              pkgs.jq
              runtimeIproute2
              iproute2BinCompat
              pkgs.coreutils
              runtimeFindutils
              pkgs.gnused
              pkgs.cacert
              continuum-proxy
            ];
            pathsToLink = [
              "/bin"
              "/lib"
              "/app"
              "/usr/bin"
              "/usr/sbin"
              "/sbin"
            ];
          };

        rootfsDev = mkRootfs { appMode = "dev"; };
        rootfsRuntimeClosure = pkgs.closureInfo { rootPaths = [ rootfsDev ]; };
        reviewedGlibcOutputs = pkgs.writeText "reviewed-glibc-output-paths" (
          pkgs.lib.concatMapStringsSep "\n" (output: toString pkgs.glibc.${output}) pkgs.glibc.outputs + "\n"
        );
        glibcNativeSmokeSource = pkgs.writeText "glibc-native-smoke.c" ''
          #define _GNU_SOURCE
          #include <dlfcn.h>
          #include <gnu/libc-version.h>
          #include <locale.h>
          #include <netdb.h>
          #include <pthread.h>
          #include <stdint.h>
          #include <stdlib.h>
          #include <string.h>
          #include <sys/random.h>
          #include <sys/types.h>
          #include <time.h>

          static void *thread_main(void *argument) {
            return argument;
          }

          int main(void) {
            unsigned char random_bytes[32];
            pthread_t thread;
            void *thread_result = NULL;
            void *libc_handle = NULL;
            struct addrinfo hints = {0};
            struct addrinfo *addresses = NULL;
            struct timespec now = {0};
            uint32_t sentinel = 0x524e4732;
            ssize_t random_length;

            if (strcmp(gnu_get_libc_version(), "${glibcUpstream.version}") != 0)
              return 10;
            random_length = getrandom(random_bytes, sizeof(random_bytes), 0);
            if (random_length < 0 || (size_t)random_length != sizeof(random_bytes))
              return 11;
            if (getentropy(random_bytes, sizeof(random_bytes)) != 0)
              return 12;
            if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
              return 13;
            if (setlocale(LC_ALL, "C") == NULL)
              return 14;
            if (pthread_create(&thread, NULL, thread_main, &sentinel) != 0)
              return 15;
            if (pthread_join(thread, &thread_result) != 0 || thread_result != &sentinel)
              return 16;

            libc_handle = dlopen("libc.so.6", RTLD_NOW | RTLD_LOCAL);
            if (libc_handle == NULL || dlsym(libc_handle, "getrandom") == NULL)
              return 17;
            if (dlclose(libc_handle) != 0)
              return 18;

            hints.ai_family = AF_INET;
            hints.ai_flags = AI_NUMERICHOST;
            if (getaddrinfo("127.0.0.1", NULL, &hints, &addresses) != 0)
              return 19;
            freeaddrinfo(addresses);
            return 0;
          }
        '';
        glibcNativeSmoke =
          pkgs.runCommandCC "glibc-native-smoke-${glibcUpstream.packageVersion}"
            {
              nativeBuildInputs = [
                pkgs.binutils
                pkgs.file
                pkgs.patchelf
              ];
            }
            ''
              "$CC" -Wall -Wextra -Werror ${glibcNativeSmokeSource} -o glibc-native-smoke -ldl -pthread
              file -b glibc-native-smoke | grep -q '^ELF '
              test "$(patchelf --print-interpreter glibc-native-smoke)" = '${pkgs.stdenv.cc.bintools.dynamicLinker}'
              readelf -lW glibc-native-smoke | grep -F 'GNU_RELRO' >/dev/null
              readelf -dW glibc-native-smoke | grep -F 'BIND_NOW' >/dev/null
              if ldd_output=$(${pkgs.glibc.bin}/bin/ldd ./glibc-native-smoke 2>&1); then
                ! grep -Fq 'not found' <<<"$ldd_output"
              else
                echo "$ldd_output" >&2
                exit 1
              fi
              ./glibc-native-smoke
              mkdir -p "$out/bin"
              install -m 0555 glibc-native-smoke "$out/bin/"
            '';
        rootfsCommandClosure =
          pkgs.runCommand "opensecret-rootfs-command-closure"
            {
              nativeBuildInputs = [
                pkgs.binutils
                pkgs.file
                pkgs.gnugrep
                pkgs.gnused
              ];
            }
            ''
              set -euo pipefail
              rootfs=${rootfsDev}
              closure_paths=${rootfsRuntimeClosure}/store-paths
              expected_interpreter=${pkgs.stdenv.cc.bintools.dynamicLinker}

              grep -Fx '${pkgs.glibc}' "$closure_paths" >/dev/null
              while IFS= read -r closure_path; do
                store_item="''${closure_path##*/}"
                package_name="''${store_item#*-}"
                case "$package_name" in
                  glibc|glibc-*|*-glibc|*-glibc-*)
                    if ! grep -Fx "$closure_path" ${reviewedGlibcOutputs} >/dev/null; then
                      echo "unreviewed glibc output in rootfs runtime closure: $closure_path" >&2
                      exit 1
                    fi
                    ;;
                esac
              done < "$closure_paths"
              if grep -Fi busybox "$closure_paths"; then
                echo "BusyBox unexpectedly re-entered the rootfs runtime closure" >&2
                exit 1
              fi

              for command in \
                bash sh base64 cat cp date find install jq ln mkdir python3 \
                sed sleep socat timeout uname continuum-proxy kmstool_enclave_cli \
                opensecret; do
                test -x "$rootfs/bin/$command"
              done
              test -x "$rootfs/bin/ip"
              test -x "$rootfs/sbin/ip"
              test "$(readlink -f "$rootfs/bin/ip")" = "$(readlink -f "$rootfs/sbin/ip")"
              test ! -e "$rootfs/bin/busybox"
              test -f "$rootfs/app/APP_MODE"
              test -f "$rootfs/app/traffic_forwarder.py"
              test -f "$rootfs/app/vsock_helper.py"

              test "$(printf rng | "$rootfs/bin/base64")" = "cm5n"
              test "$(printf rng | "$rootfs/bin/sed" -n 's/^r/R/p')" = "Rng"
              test "$("$rootfs/bin/find" "$rootfs/app" -maxdepth 1 -name APP_MODE -print -quit)" = "$rootfs/app/APP_MODE"
              "$rootfs/bin/timeout" 5 "$rootfs/bin/date" +%s >/dev/null
              "$rootfs/bin/timeout" 5 "$rootfs/bin/sleep" 0
              "$rootfs/bin/cat" "$rootfs/app/APP_MODE" >/dev/null
              "$rootfs/bin/uname" -r >/dev/null
              "$rootfs/bin/ip" -Version >/dev/null

              site_packages_count=$("$rootfs/bin/find" ${runtimePython}/lib -type d -name site-packages -print | "$rootfs/bin/wc" -l)
              test "$site_packages_count" -eq 1

              while IFS= read -r -d "" executable; do
                if ! file -b "$executable" | grep -q '^ELF '; then
                  continue
                fi

                program_headers=$(readelf -lW "$executable")
                if grep -q 'INTERP' <<<"$program_headers"; then
                  interpreter=$(sed -n 's/.*Requesting program interpreter: \(.*\)]/\1/p' <<<"$program_headers")
                  if [ "$interpreter" != "$expected_interpreter" ]; then
                    echo "unexpected ELF interpreter for $executable: $interpreter" >&2
                    exit 1
                  fi
                  if ! ldd_output=$(${pkgs.glibc.bin}/bin/ldd "$executable" 2>&1); then
                    echo "ldd failed for $executable" >&2
                    echo "$ldd_output" >&2
                    exit 1
                  fi
                  if grep -Fq 'not found' <<<"$ldd_output"; then
                    echo "missing shared library for $executable" >&2
                    echo "$ldd_output" >&2
                    exit 1
                  fi
                fi

                if grep -Eq 'GNU_STACK[[:space:]].*RWE' <<<"$program_headers"; then
                  echo "executable stack found in $executable" >&2
                  exit 1
                fi
              done < <("$rootfs/bin/find" -L "$rootfs" -type f -perm -0100 -print0)

              touch "$out"
            '';

        # Build the enclave kernel from an immutable kernel.org source pin. Nixpkgs
        # remains the build framework, but no longer controls this kernel's patch
        # level or security-update cadence.
        kernelUpstream = securityUpstreams.linux;
        expectedKernelBranch = "6.12";
        expectedKernelUrl = "https://cdn.kernel.org/pub/linux/kernel/v${pkgs.lib.versions.major kernelUpstream.version}.x/linux-${kernelUpstream.version}.tar.xz";
        kernelSource = pkgs.fetchurl {
          inherit (kernelUpstream) url hash;
        };
        kernelStructuredExtraConfig = with pkgs.lib.kernel; {
          VIRTIO_MMIO = yes;
          VIRTIO_MENU = yes;
          VIRTIO_MMIO_CMDLINE_DEVICES = yes;
          NET = yes;
          VSOCKETS = yes;
          VIRTIO_VSOCKETS = yes;
          HW_RANDOM = yes;
          NSM = yes; # Enable NSM driver for KMS operations (merged in 6.8+)
          # Disable algif_aead, the AF_ALG AEAD interface abused by CVE-2026-31431 (Copy Fail).
          CRYPTO_USER_API_AEAD = no;
        };
        unverifiedCustomKernel = pkgs.linux_6_12.override {
          argsOverride = {
            inherit (kernelUpstream) version;
            modDirVersion = kernelUpstream.version;
            src = kernelSource;
          };
          structuredExtraConfig = kernelStructuredExtraConfig;
          # Ensure we catch invalid or renamed config flags at build time
          ignoreConfigErrors = false;
        };
        customKernel =
          assert pkgs.lib.assertMsg (
            kernelUpstream.branch == expectedKernelBranch
          ) "The enclave kernel must remain on the reviewed ${expectedKernelBranch} LTS branch";
          assert pkgs.lib.assertMsg (
            pkgs.lib.versions.majorMinor kernelUpstream.version == kernelUpstream.branch
          ) "The enclave kernel version must match its declared LTS branch";
          assert pkgs.lib.assertMsg (
            kernelUpstream.url == expectedKernelUrl
          ) "The enclave kernel URL must be the version-matched kernel.org stable tarball";
          assert pkgs.lib.assertMsg (pkgs.lib.hasPrefix "sha256-" kernelUpstream.hash)
            "The enclave kernel source must use an immutable SRI sha256 hash";
          assert pkgs.lib.assertMsg (
            unverifiedCustomKernel.version == kernelUpstream.version
          ) "The realized enclave kernel version differs from the direct upstream pin";
          assert pkgs.lib.assertMsg (
            unverifiedCustomKernel.modDirVersion == kernelUpstream.version
          ) "The realized enclave kernel module version differs from the direct upstream pin";
          assert pkgs.lib.assertMsg (
            builtins.length (
              pkgs.lib.filter (
                input: (input.drvPath or null) == runtimeElfutils.drvPath
              ) unverifiedCustomKernel.nativeBuildInputs
            ) == 1
            &&
              builtins.length (
                pkgs.lib.filter (
                  input: (input.drvPath or null) == runtimeElfutils.drvPath
                ) unverifiedCustomKernel.moduleBuildDependencies
              ) == 1
            &&
              builtins.length (
                pkgs.lib.filter (input: (input.drvPath or null) == runtimeElfutils.drvPath) pkgs.pahole.buildInputs
              ) == 1
          ) "The enclave kernel toolchain is not using the reviewed elfutils derivation throughout";
          assert pkgs.lib.assertMsg (
            kernelStructuredExtraConfig.NSM == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VIRTIO_MMIO == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VIRTIO_MENU == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VIRTIO_MMIO_CMDLINE_DEVICES == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.NET == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VSOCKETS == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.VIRTIO_VSOCKETS == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.HW_RANDOM == pkgs.lib.kernel.yes
            && kernelStructuredExtraConfig.CRYPTO_USER_API_AEAD == pkgs.lib.kernel.no
          ) "The enclave kernel's required NSM, virtio, vsock, networking, or AEAD configuration changed";
          unverifiedCustomKernel;

        kernelSourcePin =
          pkgs.runCommand "opensecret-kernel-source-pin-${kernelUpstream.version}"
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

              # 6.12.101 carries both upstream NSM fixes we require: malformed
              # userspace pointers return before the mutex is acquired/unlocked,
              # and the file operations retain module ownership.
              ioctl_source="$(sed -n \
                '/static long nsm_dev_ioctl/,/static int nsm_device_init_vq/p' \
                nsm.c)"
              printf '%s\n' "$ioctl_source" \
                | grep -Fq 'if (copy_from_user(&raw, argp, _IOC_SIZE(cmd)))'
              printf '%s\n' "$ioctl_source" \
                | grep -A1 -F 'if (copy_from_user(&raw, argp, _IOC_SIZE(cmd)))' \
                | grep -Fq 'return r;'
              grep -Fq '.owner = THIS_MODULE,' nsm.c

              {
                echo 'branch=${kernelUpstream.branch}'
                echo 'version=${kernelUpstream.version}'
                echo 'source=${kernelUpstream.url}'
                echo 'hash=${kernelUpstream.hash}'
                echo 'store_path=${kernelSource}'
              } > "$out"
            '';

        kernelSecurityInvariants =
          pkgs.runCommand "opensecret-kernel-security-invariants-${kernelUpstream.version}"
            { nativeBuildInputs = [ pkgs.gnugrep ]; }
            ''
              set -euo pipefail

              config=${customKernel.configfile}
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

        opensslSourcePin =
          assert pkgs.lib.assertMsg (
            pinnedOpenSsl.version == opensslUpstream.version
          ) "The backend OpenSSL package must resolve to the direct upstream version pin";
          assert pkgs.lib.assertMsg (
            !pkgs.stdenv.isLinux
            || (
              pkgs.openssl.version == opensslUpstream.version
              && pkgs.openssl_3_5.version == opensslUpstream.version
              && pkgs.openssl.src.outPath == pkgs.openssl_3_5.src.outPath
            )
          ) "Both Linux/EIF OpenSSL package aliases must use the reviewed version and source pin";
          assert pkgs.lib.assertMsg (
            pinnedOpenSsl.src.url == opensslUpstream.url && pinnedOpenSsl.src.outputHash == opensslUpstream.hash
          ) "The realized OpenSSL source URL or hash differs from the reviewed upstream pin";
          # A source derivation is sufficient for this cross-platform check: its
          # fixed-output hash verifies the official archive without rebuilding
          # the overlay-affected Darwin bootstrap toolchain.
          pinnedOpenSsl.src;

        entrypointEntropyPreflight =
          pkgs.runCommand "opensecret-entrypoint-entropy-preflight"
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

        legacyNitroBuildRetired =
          pkgs.runCommand "opensecret-legacy-nitro-build-retired"
            {
              nativeBuildInputs = [
                pkgs.bash
                pkgs.gnugrep
                pkgs.gnused
              ];
            }
            ''
              REPO_ROOT_UNDER_TEST=${self} \
                bash ${./tests/legacy_nitro_build_retired.sh}
              touch "$out"
            '';

        # Keep the currently deployed boot behavior byte-for-byte explicit so a
        # nitro-util default change cannot silently alter entropy trust or other
        # kernel semantics. Changes to these flags require a separate staged
        # compatibility decision.
        enclaveKernelCmdline = "reboot=k panic=30 pci=off nomodules console=ttyS0 random.trust_cpu=on root=/dev/ram0";

        # Build the pinned Nitro init source locally so the package expression
        # stays compatible with current buildGoModule (`env.CGO_ENABLED`) and
        # cannot silently inherit a precompiled init blob.
        nitroInit = buildGo126Module {
          pname = "opensecret-eif-init";
          version = builtins.substring 0 12 nitroUtilUpstream.rev;
          src = "${nitro-util}/init";
          vendorHash = null;
          env.CGO_ENABLED = "0";
          ldflags = [
            "-s"
            "-w"
          ];
          postInstall = ''
            test -x "$out/bin/init"
          '';
        };

        # Function to create EIF with specific APP_MODE
        mkEif =
          {
            appMode,
            opensecretPkg ? opensecret,
            nameSuffix ? "",
          }:
          nitro.buildEif {
            name = "opensecret-eif-${appMode}${nameSuffix}";
            # The kernel image location varies by architecture
            kernel =
              if arch == "aarch64" then
                "${customKernel}/Image" # ARM64 uses Image
              else
                "${customKernel}/bzImage"; # x86_64 uses bzImage
            # EIF metadata must describe the exact configuration used to build this
            # custom kernel, rather than the unrelated pre-built Nitro blob config.
            kernelConfig = customKernel.configfile;
            cmdline = enclaveKernelCmdline;
            # NSM driver is built into kernel 6.8+, so we don't need the old module
            # Setting to null to skip loading the incompatible old module
            nsmKo = null;
            copyToRoot = mkRootfs { inherit appMode opensecretPkg; };
            entrypoint = "/bin/entrypoint";
            init = "${nitroInit}/bin/init";
          };

        opensecret = appRustPlatform.buildRustPackage {
          pname = "opensecret";
          version = "0.1.0";
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter =
              path: type:
              let
                baseName = baseNameOf path;
                parts = pkgs.lib.splitString "/" path;
              in
              # Explicitly exclude .env files
              (baseName != ".env" && baseName != ".env.sample")
              && (
                (builtins.elem "src" parts)
                || (
                  type == "regular"
                  && (baseName == "Cargo.toml" || baseName == "Cargo.lock" || baseName == "rust-toolchain.toml")
                )
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
            pinnedOpenSsl
            pkgs.zlib
            runtimePostgresql
          ];
          LIBPQ_LIB_DIR = "${runtimePostgresql.lib}/lib";
          PQ_LIB_DIR = "${runtimePostgresql.lib}/lib";
        };

        # Build the KMS/NSM helper from a fully fixed source closure instead of
        # copying checked-in ELF blobs. This is intentionally Linux-only at
        # realization time, and uses a compiler independent of the backend.
        nitro-bins = import ./nix/nitro-bins {
          inherit pkgs;
          rustToolchain = nitroRustToolchain;
          rustToolchainVersion = nitroRustToolchainVersion;
        };

        continuumProxySrc = pkgs.lib.fileset.toSource {
          root = ./privatemode-public;
          fileset = pkgs.lib.fileset.unions [
            ./privatemode-public/go.mod
            ./privatemode-public/go.sum
            ./privatemode-public/version.nix
            ./privatemode-public/privatemode-proxy/cmd
            ./privatemode-public/privatemode-proxy/internal
            ./privatemode-public/privatemode-proxy/main.go
            ./privatemode-public/internal/oss
          ];
        };
        continuum-proxy = buildGo126Module {
          pname = "continuum-proxy";
          version = continuumProxyUpstream.version;
          src = continuumProxySrc;
          vendorHash = continuumProxyUpstream.vendorHash;
          proxyVendor = true;
          doCheck = true;
          env.CGO_ENABLED = "0";
          tags = [ "contrast_unstable_api" ];
          ldflags = [
            "-s"
            "-w"
            "-X github.com/edgelesssys/continuum/internal/oss/constants.version=v${continuumProxyUpstream.version}"
          ];
          subPackages = [ "privatemode-proxy" ];
          preBuild = ''
            source_version="$(sed -n 's/.*version = "\([^"]*\)".*/\1/p' version.nix)"
            if [ "$source_version" != "v${continuumProxyUpstream.version}" ]; then
              echo "Continuum source version $source_version differs from the reviewed manifest" >&2
              exit 1
            fi
          '';
          checkPhase = ''
            runHook preCheck
            go test -count=1 -tags=contrast_unstable_api \
              ./privatemode-proxy/... \
              ./internal/oss/...
            runHook postCheck
          '';
          postInstall = ''
            mv "$out/bin/privatemode-proxy" "$out/bin/continuum-proxy"
          '';
          doInstallCheck = true;
          installCheckPhase = ''
            runHook preInstallCheck
            actual_version="$($out/bin/continuum-proxy --version)"
            expected_version="privatemode-proxy version v${continuumProxyUpstream.version}"
            if [ "$actual_version" != "$expected_version" ]; then
              echo "Unexpected Continuum proxy version output: $actual_version" >&2
              exit 1
            fi
            runHook postInstallCheck
          '';
          postFixup = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            if ! dynamic_info="$(${pkgs.binutils}/bin/readelf -d "$out/bin/continuum-proxy")"; then
              echo "failed to inspect Continuum proxy dynamic dependencies" >&2
              exit 1
            fi
            if grep -Fq NEEDED <<<"$dynamic_info"; then
              echo "continuum-proxy must remain statically linked" >&2
              exit 1
            fi
            if ! program_headers="$(${pkgs.binutils}/bin/readelf -l "$out/bin/continuum-proxy")"; then
              echo "failed to inspect Continuum proxy program headers" >&2
              exit 1
            fi
            if grep -Fq INTERP <<<"$program_headers"; then
              echo "continuum-proxy must not contain a dynamic interpreter" >&2
              exit 1
            fi
          '';
        };

        arch = pkgs.stdenv.hostPlatform.uname.processor;
        linuxGlibcBootstrapGuard =
          if !pkgs.stdenv.isLinux then
            true
          else
            let
              expectedPath = pkgs.glibc.outPath;
              expectedDynamicLinker =
                if system == "aarch64-linux" then
                  "${expectedPath}/lib/ld-linux-aarch64.so.1"
                else if system == "x86_64-linux" then
                  "${expectedPath}/lib/ld-linux-x86-64.so.2"
                else
                  null;
              libcConsumers = [
                pkgs.libc
                pkgs.stdenv.cc.libc
                pkgs.stdenv.cc.cc.stdenv.cc.libc
                runtimeBash.stdenv.cc.libc
                runtimeBash.stdenv.cc.cc.stdenv.cc.libc
              ];
              observed = map (libc: {
                inherit (libc) name outPath version;
              }) libcConsumers;
            in
            pkgs.lib.assertMsg (
              builtins.elem system [
                "aarch64-linux"
                "x86_64-linux"
              ]
              && pkgs.glibc.version == glibcUpstream.version
              && pkgs.glibc.name == "glibc-${glibcUpstream.packageVersion}"
              && pkgs.glibc.stableRev == glibcUpstream.stableRev
              && pkgs.glibc.src.url == glibcUpstream.url
              && pkgs.glibc.src.outputHash == glibcUpstream.hash
              &&
                pkgs.glibc.patchFlags == [
                  "-p1"
                  "--fuzz=0"
                  "--batch"
                  "--forward"
                ]
              && pkgs.stdenv.cc.cc.version == "15.2.0"
              && pkgs.lib.hasInfix "const char *q = strchr" (pkgs.stdenv.cc.cc.postPatch or "")
              && pkgs.lib.all (
                libc: libc.version == glibcUpstream.version && libc.outPath == expectedPath
              ) libcConsumers
              && pkgs.stdenv.cc.bintools.dynamicLinker == expectedDynamicLinker
              && runtimeBash.stdenv.cc.bintools.dynamicLinker == expectedDynamicLinker
            ) "glibc bootstrap closure differs from the reviewed 2.44 pin: ${builtins.toJSON observed}";
        runtimeVersionGuard = pkgs.lib.assertMsg (
          nixpkgs.rev == nixpkgsUpstream.rev
          && nitro-util.rev == nitroUtilUpstream.rev
          && runtimeBash.version == bashUpstream.version
          && pkgs.cacert.version == nixRuntimeUpstream.cacert
          && pkgs.coreutils.version == nixRuntimeUpstream.coreutils
          && runtimeFindutils.version == nixRuntimeUpstream.findutils
          && pkgs.gnused.version == nixRuntimeUpstream.gnused
          && runtimeGo.version == nixRuntimeUpstream.go
          && pkgs.jq.version == nixRuntimeUpstream.jq
          && runtimeKrb5.version == nixRuntimeUpstream.krb5
          && runtimePostgresql.version == nixRuntimeUpstream.postgresql
          && runtimePython.version == nixRuntimeUpstream.python
          && pkgs.socat.version == nixRuntimeUpstream.socat
          && pkgs.zlib.version == nixRuntimeUpstream.zlib
          && (
            !pkgs.stdenv.isLinux
            || (
              pkgs.glibc.version == nixRuntimeUpstream.glibc
              && runtimeElfutils.version == nixRuntimeUpstream.elfutils
              && runtimeElfutils.version == elfutilsUpstream.version
              && runtimeElfutils.src.url == elfutilsUpstream.url
              && runtimeElfutils.src.outputHash == elfutilsUpstream.hash
              && (runtimeElfutils.env.NIX_CFLAGS_COMPILE or "") == ""
              &&
                runtimeElfutils.patchFlags == [
                  "-p1"
                  "--fuzz=0"
                  "--batch"
                  "--forward"
                ]
              && builtins.length runtimeElfutils.patches == 3
              && pkgs.lib.hasInfix "debug-info-from-env.patch" (
                runtimeElfutilsPatchName (builtins.elemAt runtimeElfutils.patches 0)
              )
              && pkgs.lib.hasInfix "fix-aarch64_fregs.patch" (
                runtimeElfutilsPatchName (builtins.elemAt runtimeElfutils.patches 1)
              )
              && pkgs.lib.hasInfix "fix-i386_tlsdesc_relocation.patch" (
                runtimeElfutilsPatchName (builtins.elemAt runtimeElfutils.patches 2)
              )
              && runtimeKrb5.env.NIX_CFLAGS_COMPILE == expectedRuntimeKrb5CFlags
              && pkgs.krb5.drvPath == runtimeKrb5.drvPath
              && runtimeIproute2.version == iproute2Upstream.version
            )
          )
        ) "The Nixpkgs revision or selected EIF runtime versions changed without review";
      in
      assert runtimeVersionGuard;
      assert linuxGlibcBootstrapGuard;
      {
        packages = {
          default = opensecret;
          continuum-proxy = continuum-proxy;
        }
        // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          inherit nitro-bins nitroInit;
          glibc-native-smoke = glibcNativeSmoke;
          eif-dev = mkEif { appMode = "dev"; };
          eif-prod = mkEif { appMode = "prod"; };
          eif-preview = mkEif { appMode = "preview"; };
        };

        checks = {
          bash-runtime = runtimeBash;
          entrypoint-entropy-preflight = entrypointEntropyPreflight;
          findutils-runtime = runtimeFindutils;
          kernel-source-pin = kernelSourcePin;
          legacy-nitro-build-retired = legacyNitroBuildRetired;
          openssl-source-pin = opensslSourcePin;
          vsock-helper = vsockHelper;
        }
        // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          elfutils-runtime = runtimeElfutils;
          glibc-native-smoke = glibcNativeSmoke;
          glibc-runtime = pkgs.glibc;
          iproute2-runtime = runtimeIproute2;
          kernel-security-invariants = kernelSecurityInvariants;
          nitro-helper = nitro-bins;
          nitro-init = nitroInit;
          rootfs-command-closure = rootfsCommandClosure;
          continuum-proxy = continuum-proxy;
        };

        devShell = pkgs.mkShell {
          packages = inputs;
          shellHook = ''
            export PGDATA="''${PGDATA:-$PWD/.pgdata}"
            export PGPORT="''${PGPORT:-5432}"
            export PGSOCKETS="''${PGSOCKETS:-$PGDATA/sockets}"
            export OPENSECRET_DEV_DATABASE_URL="''${OPENSECRET_DEV_DATABASE_URL:-postgres://opensecret_user:password@localhost:$PGPORT/opensecret}"

            export LIBCLANG_PATH=${pkgs.libclang.lib}/lib/
            export LD_LIBRARY_PATH=${pinnedOpenSsl}/lib:$LD_LIBRARY_PATH
            export CC_wasm32_unknown_unknown=${pkgs.llvmPackages.clang-unwrapped}/bin/clang
            export CFLAGS_wasm32_unknown_unknown="-I ${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.lib.versions.major pkgs.llvmPackages.libclang.version}/include/"
            export LIBPQ_LIB_DIR=${runtimePostgresql.lib}/lib
            export PQ_LIB_DIR=${runtimePostgresql.lib}/lib
            export PKG_CONFIG_PATH=${pinnedOpenSsl.dev}/lib/pkgconfig:${runtimePostgresql.dev}/lib/pkgconfig:''${PKG_CONFIG_PATH:-}

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

            if ! ${setupPostgresScript}; then
              echo "Pinned PostgreSQL setup failed; refusing to enter an ambiguous dev environment." >&2
              exit 1
            fi
            if ! ${setupEnvScript}; then
              echo "Local environment setup failed." >&2
              exit 1
            fi
          '';
        };
      }
    );
}
