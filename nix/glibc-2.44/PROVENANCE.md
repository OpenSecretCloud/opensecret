# glibc 2.44 patch audit artifacts

Generated 2026-08-04 from the official GNU glibc release tarball and the
Sourceware glibc repository. These are review artifacts; they have not been
natively compiled in this macOS environment.

## Source

- Release URL: `https://ftp.gnu.org/gnu/glibc/glibc-2.44.tar.xz`
- Release SHA-256: `37f600f2bef3c5e8300147059568b2a2e40a7ad6ccc65ce942556d49429cc667`
- Release Nix SRI SHA-256: `sha256-N/YA8r7zxegwAUcFlWiyouQKetbMxlzpQlVtSUKcxmc=`
- Annotated tag object: `6c33941ff7188ac0813314f249a402a84b1a9d36`
- Tag commit: `c3a3a9808ad3ab4a3336836833f83288b672ccbf`
- Reviewed stable head: `58da792d8a2d8f2fe711318836e853fcddfd7cd8`
- Description: `glibc-2.44-8-g58da792d`

Generate `glibc-2.44-master.patch` from a checkout at the reviewed stable head:

```sh
git log --reverse --format='%H' glibc-2.44..58da792d8a2d8f2fe711318836e853fcddfd7cd8 |
while read -r rev; do
  git show --minimal "$rev" -- . ':(exclude)ADVISORIES'
done > glibc-2.44-master.patch
```

This follows the command in Nixpkgs glibc 2.43 update PR #502924 and avoids the
case-insensitive `ADVISORIES` versus `advisories/` collision.

## Artifact hashes

```text
37787c13a7168ff0e4a7e72ca0fd752c93b3507d6b09be5a1b1c47d903003e5c  glibc-2.44-master.patch
16c6ff7b052c0ea465b9e80cd0a5be9b07292927c54a99339e15a1d3fb22f6e1  dont-use-system-ld-so-cache.patch
ca877406881ec87cca96983230d316f9d499e5484d688f51f498a82ae5d99920  0001-Revert-Remove-all-usage-of-BASH-or-BASH-in-installed.patch
108f9ea7fd43b7dd77200207ddbb99f3550e7c6b169b59dbdb59ed6a72293069  0001-aarch64-math-vector.h-add-NVCC-include-guard.patch
```

The BASH and AArch64 patches are the semantic refreshes at Nixpkgs PR #502924
head `9eb03bd6fcf929bb87cfa04095ee48a8ccf01abc`. The cache patch is refreshed
again for 2.44: it routes `LD_SO_CONF`, `LD_SO_CACHE`, `TUNABLES_CONF`, and
`TUNABLES_CACHE` through the Nix prefix and adds `PREFIX` flags for
`tunconf.c`.

## Exact aarch64-linux patch order

1. `glibc-2.44-master.patch`
2. pinned `nix-locale-archive.patch`
3. refreshed `dont-use-system-ld-so-cache.patch`
4. pinned `dont-use-system-ld-so-preload.patch`
5. pinned `fix_path_attribute_in_getconf.patch`
6. pinned `nix-nss-open-files.patch`
7. refreshed `0001-Revert-Remove-all-usage-of-BASH-or-BASH-in-installed.patch`
8. pinned `reenable_DT_HASH.patch`
9. pinned `0001-localedata-allow-reproducible-parallel-install-of-lo.patch`
10. pinned `0002-Makeconfig-make-inst_complocaledir-overridable.patch`
11. refreshed `0001-aarch64-math-vector.h-add-NVCC-include-guard.patch`

Apply with `-p1 --fuzz=0`. This full sequence applied successfully to the
official 2.44 tarball. No `SYSCONFDIR` reference remained in the patched
cache/tunables files.

Drop the old `2.42-master.patch`, `fix-x64-abi.patch`, and all three extra
`resolv` CVE patches. The latter are already in 2.44. The x86 patch is
superseded by upstream's stack-aligning `sysdeps/x86_64/tls_get_addr.S`
(upstream commit `031e519c95c069abe4e4c7c59e2b4b67efccdee5`) and is irrelevant to the
AArch64 EIF.
