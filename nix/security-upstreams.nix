# Security-sensitive upstreams are pinned independently of the Nixpkgs snapshot.
# Update the version, URL, and immutable source hash together after reviewing the
# upstream stable changelog.
{
  nixpkgs = {
    branch = "nixos-26.05";
    rev = "531670d871c0e29724a02f3cbcac170adc65b58c";
  };

  # These exact versions make major-line choices visible during a Nixpkgs
  # refresh. The daily freshness job tracks the branch head; changing any
  # selected runtime remains a reviewed, dev-enclave-gated update.
  nixRuntime = {
    glibc = "2.44";
    cacert = "3.126";
    coreutils = "9.11";
    elfutils = "0.195";
    findutils = "4.11.0";
    gnused = "4.10";
    go = "1.26.5";
    jq = "1.8.2";
    krb5 = "1.22.2";
    postgresql = "17.10";
    python = "3.13.14";
    socat = "1.8.1.3";
    zlib = "1.3.2";
  };

  # glibc participates in the Linux stdenv bootstrap, so this direct pin is
  # applied at bootstrap stage 2 rather than as a leaf-package override.
  glibc = {
    version = "2.44";
    packageVersion = "2.44-8";
    stableRev = "58da792d8a2d8f2fe711318836e853fcddfd7cd8";
    url = "https://ftp.gnu.org/gnu/glibc/glibc-2.44.tar.xz";
    hash = "sha256-N/YA8r7zxegwAUcFlWiyouQKetbMxlzpQlVtSUKcxmc=";
    patches = {
      stable = {
        file = "glibc-2.44-master.patch";
        hash = "37787c13a7168ff0e4a7e72ca0fd752c93b3507d6b09be5a1b1c47d903003e5c";
      };
      cache = {
        file = "dont-use-system-ld-so-cache.patch";
        hash = "16c6ff7b052c0ea465b9e80cd0a5be9b07292927c54a99339e15a1d3fb22f6e1";
      };
      bash = {
        file = "0001-Revert-Remove-all-usage-of-BASH-or-BASH-in-installed.patch";
        hash = "ca877406881ec87cca96983230d316f9d499e5484d688f51f498a82ae5d99920";
      };
      aarch64Nvcc = {
        file = "0001-aarch64-math-vector.h-add-NVCC-include-guard.patch";
        hash = "108f9ea7fd43b7dd77200207ddbb99f3550e7c6b169b59dbdb59ed6a72293069";
      };
    };
  };

  # The pinned Nixpkgs snapshot still packages elfutils 0.194. Keep this
  # kernel build dependency on the current upstream release independently.
  elfutils = {
    version = "0.195";
    url = "https://sourceware.org/elfutils/ftp/0.195/elfutils-0.195.tar.bz2";
    hash = "sha256-N2Kf338fPcKBjhOPyiuAlBd9bC0PcB07tlClYSGNwCY=";
    patches.i386Tlsdesc = {
      url = "https://sourceware.org/git/?p=elfutils.git;a=patch;h=bfd519cc58e190544a6785d3f0a27fcfaf7d8da3";
      hash = "sha256-N7DL2FG1AWLc+hcnxGMbUl5TuieoAc9OD6gc0sbsiGI=";
    };
  };

  findutils = {
    version = "4.11.0";
    url = "https://ftp.gnu.org/gnu/findutils/findutils-4.11.0.tar.xz";
    hash = "sha256-v9GcsGzHHzNS1WfpAoTYzawCrIl3S76t8LUzsMEUMv0=";
  };

  bash = {
    branch = "5.3";
    version = "5.3p15";
    baseVersion = "5.3";
    url = "https://ftp.gnu.org/gnu/bash/bash-5.3.tar.gz";
    hash = "sha256-DVzYaWX4aaJs9k9Lcb57lvkKO6iz104n6OnZ1VUPMbo=";
    patches = [
      {
        number = "010";
        hash = "sha256-z3bxzOLqMAwYv/nwAtIfKAzJMazRfChRgRC5P+bnJWk=";
      }
      {
        number = "011";
        hash = "sha256-Apjfj16iox075D7X0mnFs8fDQt1bVwvqf2TWbcu+dTE=";
      }
      {
        number = "012";
        hash = "sha256-1xN5s5vrrtrxI0FEFOd/tFigpDua0xFllMbffKZ1RXM=";
      }
      {
        number = "013";
        hash = "sha256-BC+c2pZ+JL9CEZRGl0Qek9Bv9CtLmYYpqYobJJJ58gA=";
      }
      {
        number = "014";
        hash = "sha256-vUNgtAHThQfjWHg9ythTapnGeJ8NOlvQz7jEo0FEaWw=";
      }
      {
        number = "015";
        hash = "sha256-Vbec7uL8J/Z2fu1pfpOafrL+KijAFVa9dfGNWBAU9G4=";
      }
    ];
  };

  iproute2 = {
    version = "7.1.0";
    tag = "v7.1.0";
    url = "https://cdn.kernel.org/pub/linux/utils/net/iproute2/iproute2-7.1.0.tar.xz";
    hash = "sha256-/Z+huVgJQXFXyoPdcpV+MmG9vOiWNTy5NvgK8LM6S1w=";
  };

  linux = {
    branch = "6.12";
    version = "6.12.101";
    url = "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.12.101.tar.xz";
    hash = "sha256-DSHNEZM/SfcVG3ydu4zD/dyMir5QZDS4UP7s9B/CinY=";
  };

  appRust = {
    version = "1.97.1";
    overlayRev = "b6916ba032e02122d6ed3064f40cabe937363d43";
    overlayHash = "sha256-sZXy8mzUMi2cOGulhoW4HWAZB6JhXOAx1x8J4auZFWk=";
  };

  openssl = {
    branch = "3.5";
    version = "3.5.7";
    url = "https://github.com/openssl/openssl/releases/download/openssl-3.5.7/openssl-3.5.7.tar.gz";
    hash = "sha256-qMDSilKcpID582z1eS4s0hmEVSo8jkqhGiSqMa6smOg=";
  };

  # The helper compiler is independent of the application's toolchain. Keep
  # the overlay revision and its fetched-source hash explicit alongside the
  # stable Rust version it must provide.
  nitroRust = {
    version = "1.97.1";
    overlayRev = "b6916ba032e02122d6ed3064f40cabe937363d43";
    overlayHash = "sha256-sZXy8mzUMi2cOGulhoW4HWAZB6JhXOAx1x8J4auZFWk=";
  };

  # Upstream master b529ed6 changes only CI and a crates.io fetch workaround;
  # its flake introduces import-from-derivation and breaks our cross-system
  # evaluation gate. Runtime init/eif_build sources are byte-identical to this
  # pin. Fail freshness if upstream moves beyond the reviewed head.
  nitroUtil = {
    owner = "monzo";
    repo = "aws-nitro-util";
    branch = "master";
    rev = "7d755578b0b0b9850c0d7c4738a6c8daf3ff55c0";
    reviewedHead = "b529ed6299a49ebe362d3cf618b21d6dac4a2e48";
  };

  continuumProxy = {
    version = "1.50.0";
    tag = "v1.50.0";
    owner = "edgelesssys";
    repo = "privatemode-public";
    rev = "c7ebdb2623ab7d67464a2740dc3e8d427a9d5349";
    vendorHash = "sha256-adHo+dzpeWVnWk3VDVohZJK4C080JJRe/9XqaieMkuI=";
  };

  # These commits are the exact commits referenced by the named upstream
  # release tags. The hashes are Nix unpacked-source hashes, not archive byte
  # hashes, so fetchFromGitHub can verify the sources without trusting a tag at
  # build time.
  nitro = {
    sdkC = {
      version = "0.4.5";
      tag = "v0.4.5";
      owner = "aws";
      repo = "aws-nitro-enclaves-sdk-c";
      rev = "cd61b6187c8b20867ba4368d1ae62c5790c0269a";
      hash = "sha256-w+/D1uL5A2DLn0+b0kEcYcO3RmFL0mi/U83c4bfn+bE=";
    };
    nsmApi = {
      version = "0.5.2";
      tag = "v0.5.2";
      owner = "aws";
      repo = "aws-nitro-enclaves-nsm-api";
      rev = "1993eeb0620d35f5cefc50b17638b432325328f9";
      hash = "sha256-aG8bWZ1LRPbPy5whMC0eLJkwZyObz3yKhDr7ZBpqNUY=";
    };
    awsLc = {
      version = "5.4.0";
      tag = "v5.4.0";
      owner = "aws";
      repo = "aws-lc";
      rev = "f6acf748df0ea6157d55e640730b38d21a7751cd";
      hash = "sha256-5++GRDTi2BQKEMyvPGm/+C3jKhyB0djUbUmI43107/c=";
    };
    s2nTls = {
      version = "1.7.7";
      tag = "v1.7.7";
      owner = "aws";
      repo = "s2n-tls";
      rev = "853d1943bbbd7f782a159e77830cdaaf520d68ed";
      hash = "sha256-e9qi7I8eg4CRVdjXIHnwDnl13RN9KQBor+UOiAEm8mg=";
    };
    awsCCommon = {
      version = "0.14.4";
      tag = "v0.14.4";
      owner = "awslabs";
      repo = "aws-c-common";
      rev = "3a5e638aee99c9f1d65696f7df8c4e5d3dc5805c";
      hash = "sha256-15nidQkcaWzkRSHuqWrD980cMSHcXP9snLjrgThNSdU=";
    };
    awsCSdkutils = {
      version = "0.2.9";
      tag = "v0.2.9";
      owner = "awslabs";
      repo = "aws-c-sdkutils";
      rev = "a1cc19f53b63658f1b1400b36f199eafeeb895a6";
      hash = "sha256-VxQB0KOtzkjV6n47E5x+/EeSZbWI+nZ4M2oJk0vWvL4=";
    };
    awsCCal = {
      version = "0.9.15";
      tag = "v0.9.15";
      owner = "awslabs";
      repo = "aws-c-cal";
      rev = "8aa2a48a09f93c65d4cf06388e143a6584de6321";
      hash = "sha256-n4FWrj3ssl64zrYHA1JJhFGmXH3uUziOfdEwjvruGeE=";
    };
    awsCIo = {
      version = "0.27.5";
      tag = "v0.27.5";
      owner = "awslabs";
      repo = "aws-c-io";
      rev = "e2946c99521fa12d285c9a0829c92b1bf713922b";
      hash = "sha256-baeoEZMiEdchgOgCeVjSAbh6XfOVwDNPanDFMXKemQY=";
    };
    awsCCompression = {
      version = "0.3.2";
      tag = "v0.3.2";
      owner = "awslabs";
      repo = "aws-c-compression";
      rev = "d8264e64f698341eb03039b96b4f44702a9b3f83";
      hash = "sha256-YckyQZNk+48g5jrT4q8Clmy4LRwswKONvFbVtJxgpYQ=";
    };
    awsCHttp = {
      version = "0.11.0";
      tag = "v0.11.0";
      owner = "awslabs";
      repo = "aws-c-http";
      rev = "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4";
      hash = "sha256-SCdZfGIIHU6f0OArygZm0yY0wE6Hdx/JWvHZcK1DQOw=";
    };
    awsCAuth = {
      version = "0.10.4";
      tag = "v0.10.4";
      owner = "awslabs";
      repo = "aws-c-auth";
      rev = "4b5d524bf1a511b05e0fffe5bdc51800770b9427";
      hash = "sha256-qxZRGH+jHSrWAgKBMDdJQTn3bS23z94tgw/gO2IsSw4=";
    };
    jsonC = {
      version = "0.19";
      tag = "json-c-0.19-20260627";
      owner = "json-c";
      repo = "json-c";
      rev = "aa716cd8d663c976b99b0f30f102ee1d8ef63146";
      hash = "sha256-ZfwVOU6PJKHSj7XVZh5BUb3VJ+lHXZVMPdHh5fgrock=";
    };
  };
}
