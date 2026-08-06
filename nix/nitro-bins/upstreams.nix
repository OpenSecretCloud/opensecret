# Exact source matrix used by the currently deployed Nitro KMS/NSM helper
# generation. Revisions are peeled commits and hashes are Nix unpacked-source
# hashes. Dependency modernization belongs in a separate stacked layer.
{
  rust.version = "1.63.0";

  sdkC = {
    version = "0.4.0-8-g00c6048";
    owner = "aws";
    repo = "aws-nitro-enclaves-sdk-c";
    rev = "00c6048945a3adbb84bd269f8388282d81110499";
    hash = "sha256-tlit8ROuZ4CUNuT+jrcPLZoORG4bRPZiszMieGFvCzk=";
  };
  nsmApi = {
    version = "0.4.0";
    owner = "aws";
    repo = "aws-nitro-enclaves-nsm-api";
    rev = "4b851f3006c6fa98f23dcffb2cba03b39de9b8af";
    hash = "sha256-HxgQilQpKCRy8lmcjALIpwyAr1m1dcaUqcCL3D9JaCU=";
  };
  awsLc = {
    version = "1.12.0";
    owner = "aws";
    repo = "aws-lc";
    rev = "cb7712dfa896d32d55992e2cb13d5fa54fb77002";
    hash = "sha256-AqA0fIbBFk6SD533oTSXy9Kc7Fja8vgiiPTF00+V2MY=";
  };
  s2nTls = {
    version = "1.3.46";
    owner = "aws";
    repo = "s2n-tls";
    rev = "e954ee5dc878c5c343d35574e7d07246a1e59314";
    hash = "sha256-X+ZwM53ensCeUoxNa8cBO4KcWxWbb7iKxIRysImvKxw=";
  };
  awsCCommon = {
    version = "0.8.0";
    owner = "awslabs";
    repo = "aws-c-common";
    rev = "be35e65a6c67ae2ffd126516c9783ac1dd2e8910";
    hash = "sha256-zxV+Rf0bUrWao5ALnqgAWMVsuGfLKQbQXvw/ppDMd28=";
  };
  awsCSdkutils = {
    version = "0.1.2";
    owner = "awslabs";
    repo = "aws-c-sdkutils";
    rev = "e3c23f4aca31d9e66df25827645f72cbcbfb657a";
    hash = "sha256-G+ykP39EmI8BCeulTsZ/OSFKRzXVbEK0+mtJ3tugl5M=";
  };
  awsCCal = {
    version = "0.5.18";
    owner = "awslabs";
    repo = "aws-c-cal";
    rev = "1458c70a26877345ca28e333a092096afd410774";
    hash = "sha256-sT5ahf8MuIhqDV6RrRU+RgsLdwVUDEFWRZJpzQJOPGA=";
  };
  awsCIo = {
    version = "0.11.0";
    owner = "awslabs";
    repo = "aws-c-io";
    rev = "8f4508f5ec7d2949d5545e2b1ddcd1beb47a76a8";
    hash = "sha256-LIrAA3+Yd0lhCMQ9R4HT/ZFKm3y9iSm3h5vcn0ghiPA=";
  };
  awsCCompression = {
    version = "0.2.14";
    owner = "awslabs";
    repo = "aws-c-compression";
    rev = "5fab8bc5ab5321d86f6d153b06062419080820ec";
    hash = "sha256-bRNHjKhIpaWYToAFFUXUhyqYDmbL7SuZs2np/iH8Qzs=";
  };
  awsCHttp = {
    version = "0.7.6";
    owner = "awslabs";
    repo = "aws-c-http";
    rev = "0600662610aa871a11aebe6ed67a11997317cbef";
    hash = "sha256-pJGzGbIuz8UJkfmTQEZgXSOMuYixMezNZmgaRlcnmfg=";
  };
  awsCAuth = {
    version = "0.6.15";
    owner = "awslabs";
    repo = "aws-c-auth";
    rev = "831fa583b83574db29cbae139b42e0d7a1d1ebb8";
    hash = "sha256-oLX/evqtKbs5/WpSUBdLdKWKD8tdWwP0iRONuMCVu/E=";
  };
  jsonC = {
    version = "0.16";
    owner = "json-c";
    repo = "json-c";
    rev = "2f2ddc1f2dbca56c874e8f9c31b5b963202d80e7";
    hash = "sha256-KbnUWLgpg6/1wvXhUoYswyqDcgiwEcvgaWCPjNcX20o=";
  };
}
