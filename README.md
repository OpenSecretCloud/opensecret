# OpenSecret

This is the platform code for running OpenSecret's backend. This is intended to run on AWS Nitro inside an enclave.

## AWS Nitro Deployment

When deploying to AWS Nitro, you'll need to choose the appropriate environment:
- `dev` for development environment
- `preview` for preview/staging environment  
- `prod` for production environment

The current flake exports only those three named EIFs. Supporting another
environment requires adding and reviewing another named flake output.

Each environment has its own configuration, secrets, and infrastructure. Make sure to use the correct environment variables and AWS resources for your target environment.

### New Nix-based Deployment

The new deployment process uses Nix to create reproducible builds:

1. Nix builds the pinned Nitro KMS/NSM helper sources as part of the EIF.
   To inspect that helper closure independently on Linux ARM:
```bash
just build-nitro-bins
```

2. Build the EIF for your target environment:
```bash
# For development
nix build '.?submodules=1#eif-dev'

# For production
nix build '.?submodules=1#eif-prod'

# For preview
nix build '.?submodules=1#eif-preview'
```

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

## Nitro Enclaves Setup

The project uses AWS Nitro Enclaves and requires two helper artifacts:
- `libnsm.so` - NSM (Nitro Security Module) library
- `kmstool_enclave_cli` - KMS tool for key operations

Nix builds both from fixed commits and hashes declared in
`nix/nitro-bins/upstreams.nix`. The source repositories are:
- [aws-nitro-enclaves-nsm-api](https://github.com/aws/aws-nitro-enclaves-nsm-api)
- [aws-nitro-enclaves-sdk-c](https://github.com/aws/aws-nitro-enclaves-sdk-c)

### Building Nitro Binaries

To build the same helper derivation consumed by the EIF:

```bash
just build-nitro-bins
```

The build fetches only the reviewed, immutable sources and produces a Nix
result; it never writes ELF binaries back into the repository. Normal EIF
builds consume this derivation automatically.

## Building and Deploying with Nix

### Building the EIF

1. Build an explicit EIF target using Nix; its helper closure is built
   automatically (development shown):
```bash
nix build '.?submodules=1#eif-dev'
```

This will create a symlink `result` pointing to the built EIF file.

### EIF construction contract

The named Nix outputs are the only supported way to assemble the OpenSecret
application root filesystem and EIF. The parent-instance credential requester
and logging containers are operational services, not EIF build inputs.

The Nix-based build:
- Creates a more reproducible build environment
- Source-builds Nitro helper artifacts from reviewed commits and hashes
- Integrates with the Monzo aws-nitro-util for EIF creation
- Preserves the application-facing helper CLI contract

The resulting EIF is deployed with the environment-specific commands above.

## CI/CD Requirements

### GitHub Actions Runner

This project requires a custom GitHub Actions runner with the following specifications:

- Label: `ubuntu-22.04-arm64-4core`
- Architecture: ARM64
- Operating System: Ubuntu 22.04
- Resources: 4 CPU cores

The workflow uses this custom runner for both development and production builds. For more information about setting up custom GitHub Actions runners, see [GitHub's documentation](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners).
