# OpenSecret

This is the platform code for running OpenSecret's backend. This is intended to run on AWS Nitro inside an enclave.

## AWS Nitro Deployment

When deploying to AWS Nitro, you'll need to choose the appropriate environment:
- `dev` for development environment
- `preview` for preview/staging environment  
- `prod` for production environment
- `custom` for custom environment (requires `ENV_NAME` to be set)

Each environment has its own configuration, secrets, and infrastructure. Make sure to use the correct environment variables and AWS resources for your target environment.

### New Nix-based Deployment

The new deployment process uses Nix to create reproducible builds:

1. Optionally build the pinned Nitro helper closure on Linux as a standalone check:
```bash
just build-nitro-bins
```

The EIF build performs this source build automatically; it never consumes the
checked-in legacy helper blobs.

2. Build the EIF for your target environment:
```bash
# For development
nix build '.?submodules=1#eif-dev'

# For production
nix build '.?submodules=1#eif-prod'

# For preview
nix build '.?submodules=1#eif-preview'

```

Custom EIFs are not exported by the current flake. Add a named, reviewed output
for a custom environment and validate it in the dev enclave before use; do not
fall back to the retired Docker path.

This will create a symlink `result` pointing to the built EIF file.

3. Copy the EIF to your AWS parent instance:
```bash
# For development
just scp-eif-to-aws-dev

# For production
just scp-eif-to-aws-prod

# For preview
just scp-eif-to-aws-preview
```

4. Deploy the EIF:
```bash
# For development
just deploy-dev-nix

# For production
just deploy-prod-nix

# For preview
just deploy-preview-nix
```

The deployment process will:
1. Build the EIF
2. Copy it to the AWS parent instance
3. Prompt you to review the PCR values
4. After confirmation, terminate any existing enclave
5. Run the new enclave
6. Restart the socat proxy

### PCR Value Management

The Nix build process generates PCR (Platform Configuration Register) values that are used by AWS KMS for attestation. You can:

1. Copy PCR values to a reference file:
```bash
just copy-pcr-dev    # For development
just copy-pcr-prod   # For production
just copy-pcr-preview # For preview
```

2. Verify PCR values match the reference:
```bash
just verify-pcr-dev    # For development
just verify-pcr-prod   # For production
just verify-pcr-preview # For preview
```

This ensures the build is reproducible and matches the expected configuration.

### Deprecated Docker-based Deployment

Docker-based enclave deployment is retired. There is no root `Dockerfile`, the
local image definition rejects every `APP_MODE` except `local`, and the legacy
enclave-base recipe fails closed. Build all staging and production EIFs with
the Nix targets above.

## Nitro Enclaves Setup

The project uses AWS Nitro Enclaves and builds two helper artifacts from pinned
upstream sources as part of the Nix closure:
- `libnsm.so` - NSM (Nitro Security Module) library
- `kmstool_enclave_cli` - KMS tool for key operations

These binaries are built from the official AWS repositories:
- [aws-nitro-enclaves-nsm-api](https://github.com/aws/aws-nitro-enclaves-nsm-api)
- [aws-nitro-enclaves-sdk-c](https://github.com/aws/aws-nitro-enclaves-sdk-c)

### Building Nitro Binaries

Nix fetches every helper dependency by immutable commit and hash, then builds
the crypto/TLS/CRT closure without build-time network access. To build the
helper closure independently on aarch64 Linux:

```bash
just build-nitro-bins
```

This is an optional validation target. EIF builds consume the same derivation
directly and do not copy or overwrite repository binaries.

## Building and Deploying with Nix

### Building the EIF

1. Optionally build the Nitro helper closure independently:
```bash
just build-nitro-bins
```

2. Build an explicit EIF target using Nix (development shown):
```bash
nix build '.?submodules=1#eif-dev'
```

This will create a symlink `result` pointing to the built EIF file.

### Differences from Docker-based Build

The Nix-based build:
- Creates a more reproducible build environment
- Builds Nitro helper binaries from immutable, directly pinned upstream sources
- Integrates with the Monzo aws-nitro-util for EIF creation
- Produces the same functionality as the Docker-based build

The resulting EIF can be deployed and managed exactly like the Docker-built version.

## CI/CD Requirements

### GitHub Actions Runner

This project requires a custom GitHub Actions runner with the following specifications:

- Label: `ubuntu-22.04-arm64-4core`
- Architecture: ARM64
- Operating System: Ubuntu 22.04
- Resources: 4 CPU cores

The workflow uses this custom runner for both development and production builds. For more information about setting up custom GitHub Actions runners, see [GitHub's documentation](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners).


## Development

This project can be built and run using Docker. Follow these steps to build and run the Docker container:

### Building the Docker Image

1. Ensure you have Docker installed on your system.
2. Navigate to the project root directory in your terminal.

3. `Dockerfile.local` remains a local application-development artifact; it is
   not an enclave release path, contains no NSM/KMS helper, and launches the
   backend directly. Release and staging EIFs must use the Nix targets
   described above.

4. Build the local-only application image:

```sh
just build-docker-local
```

This command builds the local image and tags it as `opensecret`. The build
rejects `dev`, `preview`, `prod`, and `custom`; those are enclave modes and must
go through Nix.

### Running the Docker Container

After building the image, you can run the container using:

```sh
docker run -p 3000:3000 -p 5000:5000 --name opensecret-container opensecret
```

This command starts a new container from the `opensecret` image and maps port 3000 on the host machine to port 3000 in the container.

```sh
sh
docker run -p 3000:3000 -p 5000:5000 --name opensecret-container opensecret
```

To stop the container, use:

```sh
docker stop opensecret-container
```

To remove the container, use:

```sh
docker rm opensecret-container
```
