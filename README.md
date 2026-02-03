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

1. First, build the required Nitro binaries (only needed once):
```bash
just build-nitro-bins
```

2. Build the EIF for your target environment:
```bash
# For development
nix build .#eif-dev

# For production
nix build .#eif-prod

# For preview
nix build .#eif-preview

# For custom environments
ENV_NAME=your_env_name nix build .#eif
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

### Deprecated Docker-based Deployment

This method is deprecated as it does not provide reproducible builds. Here are the raw commands for reference:

```sh
# Build the Docker image
docker build -t opensecret --build-arg APP_MODE=dev .

# Save the image to a tar file
docker save -o opensecret.tar opensecret

# Copy to AWS parent instance
scp opensecret.tar ec2-user@[aws-parent-instance-ip]:~/

# Load the image on the parent instance
ssh ec2-user@[aws-parent-instance-ip]
docker load -i opensecret.tar && docker tag localhost/opensecret:latest opensecret:latest

# Build the EIF file
nitro-cli build-enclave --docker-uri opensecret:latest --output-file opensecret.eif

# Run the EIF file
nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4

# Or run in debug mode
nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4 --debug-mode
```

## Nitro Enclaves Setup

The project uses AWS Nitro Enclaves and requires two pre-built binaries:
- `libnsm.so` - NSM (Nitro Security Module) library
- `kmstool_enclave_cli` - KMS tool for key operations

These binaries are built from the official AWS repositories:
- [aws-nitro-enclaves-nsm-api](https://github.com/aws/aws-nitro-enclaves-nsm-api)
- [aws-nitro-enclaves-sdk-c](https://github.com/aws/aws-nitro-enclaves-sdk-c)

### Building Nitro Binaries

The binaries are built using Docker to ensure a consistent build environment. To build them:

```bash
just build-nitro-bins
```

This will:
1. Create a `nitro-bins` directory
2. Build the binaries in an Amazon Linux 2 container
3. Extract them to the `nitro-bins` directory

You only need to do this once, or when you want to update the binaries to a new version.
The built binaries are used by the Nix build process to create the EIF (Enclave Image Format).

## Building and Deploying with Nix

### Building the EIF

1. First, build the required Nitro binaries (only needed once):
```bash
just build-nitro-bins
```

2. Build the EIF using Nix:
```bash
nix build .#eif
```

This will create a symlink `result` pointing to the built EIF file.

### Differences from Docker-based Build

The Nix-based build:
- Creates a more reproducible build environment
- Uses pre-built Nitro binaries for consistency
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

3. Build the enclave base image:

```sh
docker build ./nitro-toolkit/enclave-base-image/ -t enclave_base
```

4. Build the main Docker image using the following command:

DEV:

```sh
docker build -t opensecret \
--build-arg DATABASE_URL={PROD_DB_CONNECTION} \
--build-arg OPENAI_API_KEY={YOUR_OPENAI_API_KEY} \
--build-arg APP_MODE=local \
.
```

If building for the nitro image (use `dev` [default], `preview`, `prod`, or `custom` depending on the env):

```sh
docker rmi opensecret:latest && docker build -t opensecret \
--build-arg APP_MODE=dev \
.
```

```sh
docker rmi opensecret:latest && docker build -t opensecret \
--build-arg APP_MODE=preview \
.
```

```sh
docker rmi opensecret:latest && docker build -t opensecret \
--build-arg APP_MODE=prod \
.
```

For custom environments, you must also provide an `ENV_NAME`:
```sh
docker rmi opensecret:latest && docker build -t opensecret \
--build-arg APP_MODE=custom \
--build-arg ENV_NAME=your_env_name \
.
```

This command builds the Docker image and tags it as `opensecret`. The `--build-arg` flags are used to pass the environment variables to the Docker build process:
- `DATABASE_URL`: Your production database connection string
- `OPENAI_API_KEY`: Your OpenAI API key
- `APP_MODE`: The deployment environment (`dev`, `preview`, `prod`, or `custom`)
- `ENV_NAME`: Required when `APP_MODE` is `custom`, specifies the custom environment name

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

