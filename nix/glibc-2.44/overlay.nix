{
  expectedNixpkgsRev ? "531670d871c0e29724a02f3cbcac170adc65b58c",
  actualNixpkgsRev,
  patchDir ? ./.,
  upstream,
}:

final: prev:

let
  lib = prev.lib;
  stageName = prev.stdenv.name;
  isGlibcBootstrapStage =
    prev.stdenv.hostPlatform.isLinux && stageName == "bootstrap-stage2-stdenv-linux";
  isGccBootstrapStage =
    prev.stdenv.hostPlatform.isLinux && stageName == "bootstrap-stage3-stdenv-linux";

  checkedPatch =
    path: expected:
    assert lib.assertMsg (
      builtins.hashFile "sha256" path == expected
    ) "glibc patch hash differs from the reviewed artifact: ${toString path}";
    path;

  stablePatch = checkedPatch (
    patchDir + "/${upstream.patches.stable.file}"
  ) upstream.patches.stable.hash;
  stablePatchCommits = map (line: lib.removePrefix "commit " line) (
    lib.filter (line: lib.hasPrefix "commit " line) (
      lib.splitString "\n" (builtins.readFile stablePatch)
    )
  );

  replacements = {
    "dont-use-system-ld-so-cache.patch" = checkedPatch (
      patchDir + "/${upstream.patches.cache.file}"
    ) upstream.patches.cache.hash;

    "0001-Revert-Remove-all-usage-of-BASH-or-BASH-in-installed.patch" = checkedPatch (
      patchDir + "/${upstream.patches.bash.file}"
    ) upstream.patches.bash.hash;

    "0001-aarch64-math-vector.h-add-NVCC-include-guard.patch" = checkedPatch (
      patchDir + "/${upstream.patches.aarch64Nvcc.file}"
    ) upstream.patches.aarch64Nvcc.hash;
  };

  droppedPatches = [
    "2.42-master.patch"
    "fix-x64-abi.patch"
    "0001-resolv-Check-for-inet_ntop-failure-in-ns_sprintrrf.patch"
    "0002-resolv-More-types-as-unknown-in-ns_sprintrrf-CVE-202.patch"
    "0003-resolv-Fix-buffer-overreads-in-ns_sprintrrf-CVE-2026.patch"
  ];

  nameOf = patch: builtins.baseNameOf (toString patch);

  mkGlibc244 =
    let
      oldPatches = prev.glibc.patches or [ ];
      oldNames = map nameOf oldPatches;
      needsAarchPatch = prev.stdenv.buildPlatform.isAarch64 || prev.stdenv.hostPlatform.isAarch64;
      requiredOldNames =
        droppedPatches
        ++ [
          "dont-use-system-ld-so-cache.patch"
          "0001-Revert-Remove-all-usage-of-BASH-or-BASH-in-installed.patch"
        ]
        ++ lib.optional needsAarchPatch "0001-aarch64-math-vector.h-add-NVCC-include-guard.patch";
      rewrite =
        patch:
        let
          name = nameOf patch;
        in
        if builtins.hasAttr name replacements then replacements.${name} else patch;
    in
    assert lib.assertMsg (
      actualNixpkgsRev == expectedNixpkgsRev
    ) "The glibc bootstrap overlay is coupled to the reviewed Nixpkgs revision";
    assert lib.assertMsg (
      upstream.version == "2.44"
      && upstream.packageVersion == "2.44-8"
      && upstream.patches.stable.file == "glibc-2.44-master.patch"
      && upstream.patches.cache.file == "dont-use-system-ld-so-cache.patch"
      && upstream.patches.bash.file == "0001-Revert-Remove-all-usage-of-BASH-or-BASH-in-installed.patch"
      && upstream.patches.aarch64Nvcc.file == "0001-aarch64-math-vector.h-add-NVCC-include-guard.patch"
    ) "The glibc manifest moved outside the reviewed 2.44 bootstrap design";
    assert lib.assertMsg (
      stablePatchCommits != [ ]
      && lib.all (commit: builtins.match "[0-9a-f]{40}" commit != null) stablePatchCommits
      && lib.last stablePatchCommits == upstream.stableRev
      && upstream.packageVersion == "${upstream.version}-${toString (builtins.length stablePatchCommits)}"
    ) "The glibc stable revision/package patch level is not coupled to the reviewed stable patch";
    assert lib.assertMsg (lib.all (
      name: builtins.elem name oldNames
    ) requiredOldNames) "Pinned Nixpkgs glibc patch set changed; re-audit the 2.44 override";
    prev.glibc.overrideAttrs (old: {
      # The package name records the reviewed stable-branch patch level, while
      # passthru.version remains the upstream ABI release used by Nixpkgs.
      version = upstream.packageVersion;
      src = prev.stdenv.fetchurlBoot {
        inherit (upstream) url hash;
      };
      patches = [
        stablePatch
      ]
      ++ map rewrite (lib.filter (patch: !(builtins.elem (nameOf patch) droppedPatches)) oldPatches);
      # Reject reversed/already-applied patches and never prompt during builds.
      patchFlags = [
        "-p1"
        "--fuzz=0"
        "--batch"
        "--forward"
      ];
      passthru = (old.passthru or { }) // {
        inherit (upstream) stableRev version;
        minorRelease = upstream.version;
      };
      meta = (old.meta or { }) // {
        identifiers = (old.meta.identifiers or { }) // {
          cpeParts = lib.meta.cpeFullVersionWithVendor "gnu" upstream.version;
        };
      };
    });

  gcc15ForGlibc243 =
    assert lib.assertMsg (
      prev.gcc-unwrapped.version == "15.2.0"
    ) "The glibc C23 compatibility patch is only reviewed for GCC 15.2.0";
    prev.gcc-unwrapped.overrideAttrs (old: {
      postPatch = (old.postPatch or "") + ''
        substituteInPlace libgomp/affinity-fmt.c \
          --replace-fail 'char *q = strchr' 'const char *q = strchr'
      '';
    });
in

lib.optionalAttrs isGlibcBootstrapStage {
  glibc = mkGlibc244;
}
// lib.optionalAttrs isGccBootstrapStage {
  gcc-unwrapped = gcc15ForGlibc243;
}
