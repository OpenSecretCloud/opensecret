{
  pkgs,
  # NSM API v0.4.0 declares Rust 1.63 as its MSRV. Pass an explicitly reviewed,
  # immutable toolchain and version here; this is intentionally separate from
  # the application's Rust toolchain so helper freshness cannot silently move
  # application key logic.
  rustToolchain,
  rustToolchainVersion,
}:

let
  inherit (pkgs) lib;
  # Preserve the exact source matrix used to produce the currently deployed
  # helper generation. Dependency modernization belongs in the next layer.
  upstreams = import ./upstreams.nix;

  fetchSource =
    source:
    pkgs.fetchFromGitHub {
      inherit (source)
        owner
        repo
        rev
        hash
        ;
    };

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };

  staticCmakeFlags = [
    "-GNinja"
    "-DBUILD_SHARED_LIBS=OFF"
    "-DBUILD_TESTING=OFF"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
  ];

  mkStaticCmakePackage =
    {
      source,
      buildInputs ? [ ],
      extraCmakeFlags ? [ ],
      env ? { },
      patches ? [ ],
    }:
    pkgs.stdenv.mkDerivation {
      pname = source.repo;
      inherit (source) version;
      src = fetchSource source;

      strictDeps = true;
      nativeBuildInputs = [
        pkgs.cmake
        pkgs.ninja
        pkgs.pkg-config
      ];
      inherit
        buildInputs
        env
        patches
        ;

      cmakeFlags = staticCmakeFlags ++ extraCmakeFlags;
      doCheck = false;
    };

  awsLc = mkStaticCmakePackage {
    source = upstreams.awsLc;
    extraCmakeFlags = [
      "-DBUILD_TOOL=OFF"
      "-DDISABLE_GO=ON"
      "-DDISABLE_PERL=ON"
      "-DENABLE_PRE_SONAME_BUILD=ON"
    ];
    env.NIX_CFLAGS_COMPILE = lib.optionalString pkgs.stdenv.cc.isGNU "-Wno-error=stringop-overflow";
  };

  s2nTls = mkStaticCmakePackage {
    source = upstreams.s2nTls;
    buildInputs = [ awsLc ];
    extraCmakeFlags = [
      "-DSEARCH_LIBCRYPTO=ON"
      "-DS2N_USE_CRYPTO_SHARED_LIBS=OFF"
    ];
  };

  # aws-c-common installs CMake modules that its consumers include directly.
  # Match the Nixpkgs integration so those modules remain discoverable under
  # strict dependency propagation, including in native aarch64-linux builds.
  awsCCommon = (mkStaticCmakePackage {
    source = upstreams.awsCCommon;
  }).overrideAttrs {
    setupHook = ./aws-c-common-setup-hook.sh;
  };

  awsCSdkutils = mkStaticCmakePackage {
    source = upstreams.awsCSdkutils;
    buildInputs = [ awsCCommon ];
  };

  awsCCal = mkStaticCmakePackage {
    source = upstreams.awsCCal;
    buildInputs = [
      awsCCommon
      awsLc
    ];
  };

  awsCIo = mkStaticCmakePackage {
    source = upstreams.awsCIo;
    buildInputs = [
      awsCCommon
      awsCCal
      awsLc
      s2nTls
    ];
    extraCmakeFlags = [ "-DUSE_VSOCK=ON" ];
  };

  awsCCompression = mkStaticCmakePackage {
    source = upstreams.awsCCompression;
    buildInputs = [ awsCCommon ];
  };

  awsCHttp = mkStaticCmakePackage {
    source = upstreams.awsCHttp;
    buildInputs = [
      awsCCommon
      awsCCal
      awsCIo
      awsCCompression
      awsLc
      s2nTls
    ];
    # The retired Docker matrix pairs aws-c-http 0.7.6 with aws-c-common
    # 0.8.0. Its unused websocket decoder references UTF-8 APIs introduced
    # later; GCC 11 built that dead archive member with warnings, while GCC 14
    # promotes the same diagnostics to errors. Keep this compatibility shim
    # scoped to the legacy HTTP archive and prove below that the unresolved
    # decoder API is not linked into the deployed kmstool executable.
    env.NIX_CFLAGS_COMPILE = lib.optionalString pkgs.stdenv.cc.isGNU ''
      -Wno-error=implicit-function-declaration
      -Wno-error=int-conversion
    '';
  };

  awsCAuth = mkStaticCmakePackage {
    source = upstreams.awsCAuth;
    # Backport the one-line upstream type correction from aws-c-auth 92038d9.
    # It adds the const qualifier already required by HTTP 0.7.6 and changes
    # no function ABI or runtime control flow.
    patches = [ ./aws-c-auth-const-connection-manager-options.patch ];
    buildInputs = [
      awsCCommon
      awsCSdkutils
      awsCCal
      awsCIo
      awsCCompression
      awsCHttp
      awsLc
      s2nTls
    ];
  };

  jsonC = mkStaticCmakePackage {
    source = upstreams.jsonC;
    extraCmakeFlags = [ "-DBUILD_APPS=OFF" ];
  };

  nsmApi = rustPlatform.buildRustPackage {
    pname = upstreams.nsmApi.repo;
    inherit (upstreams.nsmApi) version;
    src = fetchSource upstreams.nsmApi;

    strictDeps = true;
    cargoLock.lockFile = ./nsm-api-v0.4.0.Cargo.lock;
    postUnpack = ''
      cp ${./nsm-api-v0.4.0.Cargo.lock} "$sourceRoot/Cargo.lock"
    '';
    cargoBuildFlags = [
      "--package"
      "nsm-lib"
    ];
    cargoTestFlags = [
      "--package"
      "nsm-lib"
    ];

    preBuild = ''
      actual_rust_version="$(${rustToolchain}/bin/rustc --version | awk '{ print $2 }')"
      if [[ "$actual_rust_version" != "${rustToolchainVersion}" ]]; then
        echo "NSM API requires the explicitly reviewed Rust ${rustToolchainVersion} toolchain; got $actual_rust_version" >&2
        exit 1
      fi
    '';

    doCheck = true;

    installPhase = ''
      runHook preInstall

      release_dir="target/${pkgs.stdenv.hostPlatform.rust.rustcTarget}/release"
      test -d "$release_dir"
      mkdir -p "$out/lib" "$out/include"
      install -m 755 "$release_dir/libnsm.so" "$out/lib/libnsm.so"
      install -m 644 "$release_dir/libnsm.a" "$out/lib/libnsm.a"
      install -m 644 "$release_dir/nsm.h" "$out/include/nsm.h"

      runHook postInstall
    '';
  };

  sdkC = pkgs.stdenv.mkDerivation {
    pname = upstreams.sdkC.repo;
    inherit (upstreams.sdkC) version;
    src = fetchSource upstreams.sdkC;

    patches = [ ./sdk-c-explicit-version.patch ];
    strictDeps = true;

    nativeBuildInputs = [
      pkgs.binutils
      pkgs.cmake
      pkgs.ninja
      pkgs.patchelf
      pkgs.pkg-config
    ];
    buildInputs = [
      awsLc
      s2nTls
      awsCCommon
      awsCSdkutils
      awsCCal
      awsCIo
      awsCCompression
      awsCHttp
      awsCAuth
      jsonC
      nsmApi
    ];

    cmakeFlags = [
      "-GNinja"
      "-DBUILD_SHARED_LIBS=OFF"
      "-DBUILD_TESTING=ON"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
      "-DVERSION=v${upstreams.sdkC.version}"
    ];

    postPatch = ''
      source_file="bin/kmstool-enclave-cli/main.c"
      grep -Fq 'AWS_ASSERT(aws_nitro_enclaves_library_seed_entropy(1024) == AWS_OP_SUCCESS)' "$source_file"
      grep -Fq 'if (NOT DEFINED VERSION)' CMakeLists.txt
    '';

    doCheck = true;
    checkPhase = ''
      runHook preCheck

      # These two legacy REST-client integration tests require both an FHS CA
      # path under /etc and external KMS networking. Neither is available in a
      # hermetic Nix build sandbox, and the retired Docker build did not run
      # CTest. Keep the exclusion exact while still running the other SDK tests.
      registered_tests="$(ctest -N)"
      grep -Eq 'Test +#[0-9]+: test_basic_rest_client$' <<<"$registered_tests"
      grep -Eq 'Test +#[0-9]+: test_rest_call_blocking$' <<<"$registered_tests"
      ctest --output-on-failure \
        --exclude-regex '^(test_basic_rest_client|test_rest_call_blocking)$'

      runHook postCheck
    '';

    postInstall = ''
      cp -a ${nsmApi}/lib/libnsm.so "$out/lib/"

      # Rust's libnsm uses the compiler unwind runtime. Bundle it so the helper
      # closure is complete when the final package's lib directory is copied
      # into the enclave rootfs.
      compiler_runtime="${lib.getLib pkgs.stdenv.cc.cc}/lib/libgcc_s.so.1"
      if [[ ! -e "$compiler_runtime" ]]; then
        compiler_runtime="${lib.getLib pkgs.stdenv.cc.cc}/lib64/libgcc_s.so.1"
      fi
      if [[ ! -e "$compiler_runtime" ]]; then
        echo "could not locate libgcc_s.so.1 for libnsm" >&2
        exit 1
      fi
      cp -L "$compiler_runtime" "$out/lib/libgcc_s.so.1"
    '';

    postFixup = ''
      kmstool="$out/bin/kmstool_enclave_cli"
      nsm_real="$out/lib/libnsm.so"

      ${pkgs.patchelf}/bin/patchelf --set-rpath '$ORIGIN/../lib' "$kmstool"
      ${pkgs.patchelf}/bin/patchelf --set-rpath '$ORIGIN' "$nsm_real"

      if ! nsm_dynamic="$(${pkgs.binutils}/bin/readelf -d "$nsm_real")"; then
        echo "failed to inspect libnsm dynamic metadata" >&2
        exit 1
      fi
      if grep -Fq 'Library soname:' <<<"$nsm_dynamic"; then
        echo "legacy NSM v0.4.0 unexpectedly acquired a SONAME" >&2
        exit 1
      fi

      # These strings are the application-facing stdout contract consumed by
      # src/encrypt.rs. They must not drift during dependency-only updates.
      strings_file="$(mktemp)"
      if ! ${pkgs.binutils}/bin/strings "$kmstool" > "$strings_file"; then
        echo "failed to inspect kmstool strings" >&2
        exit 1
      fi
      grep -Fxq 'PLAINTEXT: %s' "$strings_file"
      grep -Fxq 'CIPHERTEXT: %s' "$strings_file"
      grep -Fxq 'aws-nitro_enclaves-sdk-c/v${upstreams.sdkC.version}' "$strings_file"

      if ! kmstool_dynamic="$(${pkgs.binutils}/bin/readelf -d "$kmstool")"; then
        echo "failed to inspect kmstool dynamic metadata" >&2
        exit 1
      fi
      if ! kmstool_interpreter="$(${pkgs.patchelf}/bin/patchelf --print-interpreter "$kmstool")"; then
        echo "failed to inspect kmstool ELF interpreter" >&2
        exit 1
      fi
      interpreter_soname="''${kmstool_interpreter##*/}"
      if [[ -z "$interpreter_soname" ]]; then
        echo "kmstool has an empty ELF interpreter" >&2
        exit 1
      fi

      needed="$(sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' <<<"$kmstool_dynamic")"
      if ! grep -Fxq 'libnsm.so' <<<"$needed"; then
        echo "kmstool is not linked against the legacy NSM ABI" >&2
        printf '%s\n' "$needed" >&2
        exit 1
      fi

      # Nix's glibc linker script records the target ELF loader itself as a
      # DT_NEEDED entry. It is already required as the program interpreter and
      # present in the rootfs, so permit only that exact interpreter basename.
      unexpected="$(printf '%s\n' "$needed" \
        | grep -Fvx "$interpreter_soname" \
        | grep -Ev '^(libnsm\.so|libc\.so\.6|libm\.so\.6|libgcc_s\.so\.1|libpthread\.so\.0|libdl\.so\.2|librt\.so\.1)$' \
        || true)"
      if [[ -n "$unexpected" ]]; then
        echo "unexpected dynamic dependency in kmstool; CRT/TLS/crypto dependencies must be static" >&2
        printf '%s\n' "$unexpected" >&2
        exit 1
      fi

      if grep -Eq 'lib(aws|s2n|crypto|ssl|json-c)' <<<"$needed"; then
        echo "kmstool contains a dynamically linked CRT/TLS/crypto dependency" >&2
        printf '%s\n' "$needed" >&2
        exit 1
      fi

      if ! kmstool_symbols="$(${pkgs.binutils}/bin/nm "$kmstool")"; then
        echo "failed to inspect kmstool symbols" >&2
        exit 1
      fi
      if grep -Eq 'aws_(utf8_decoder|websocket_decoder)_' <<<"$kmstool_symbols"; then
        echo "legacy HTTP websocket decoder leaked into the deployed kmstool" >&2
        exit 1
      fi

      nsm_needed="$(sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' <<<"$nsm_dynamic")"
      nsm_unexpected="$(printf '%s\n' "$nsm_needed" \
        | grep -Fvx "$interpreter_soname" \
        | grep -Ev '^(libc\.so\.6|libm\.so\.6|libgcc_s\.so\.1|libpthread\.so\.0|libdl\.so\.2|librt\.so\.1)$' \
        || true)"
      if [[ -n "$nsm_unexpected" ]]; then
        echo "unexpected dynamic dependency in libnsm; add it to the enclave rootfs or link it statically" >&2
        printf '%s\n' "$nsm_unexpected" >&2
        exit 1
      fi
      if grep -Fxq 'libgcc_s.so.1' <<<"$nsm_needed"; then
        if [[ ! -f "$out/lib/libgcc_s.so.1" ]]; then
          echo "libnsm requires libgcc_s.so.1, but it was not bundled" >&2
          exit 1
        fi
      fi
    '';

    passthru = {
      inherit
        awsLc
        s2nTls
        awsCCommon
        awsCSdkutils
        awsCCal
        awsCIo
        awsCCompression
        awsCHttp
        awsCAuth
        jsonC
        nsmApi
        ;
      sources = upstreams;
    };

    meta = {
      description = "Source-built AWS Nitro Enclaves NSM and KMS helper closure";
      homepage = "https://github.com/aws/aws-nitro-enclaves-sdk-c";
      license = lib.licenses.asl20;
      platforms = lib.platforms.linux;
      mainProgram = "kmstool_enclave_cli";
    };
  };
in
sdkC
