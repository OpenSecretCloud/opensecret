# Load environment variables from .env file
set dotenv-load

# Set the container runtime (docker or podman)
container := "podman"

# Set the default recipe to list all available commands
default:
    @just --list

# Build the enclave base image
build-enclave-base:
    {{container}} build ./nitro-toolkit/enclave-base-image/ -t enclave_base

# Build Nitro binaries from enclave base image (NSM and KMS tools)
build-nitro-bins:
    nix run .#write-nitro-bins

# Build the main Docker image for local
build-docker-local:
    {{container}} rmi opensecret:latest || true
    {{container}} build -t opensecret \
    --build-arg APP_MODE=local \
    .

### Credential Requester Commands ###

# Build the Credential Requester Docker image for development
build-credential-requester-docker:
    {{container}} rmi credential-requester:latest || true
    cd nitro-toolkit/credential_requester && \
    {{container}} build -t credential-requester .

# Save Credential Requester Docker image to a tar file for dev mode
save-credential-requester-docker-image-dev:
    rm -f build/credential-requester/dev/credential-requester.tar && \
    {{container}} save -o build/credential-requester/dev/credential-requester.tar credential-requester

# Save Credential Requester Docker image to a tar file for prod
save-credential-requester-docker-image-prod:
    rm -f build/credential-requester/prod/credential-requester.tar && \
    {{container}} save -o build/credential-requester/prod/credential-requester.tar credential-requester

# Save Credential Requester Docker image to a tar file for preview mode
save-credential-requester-docker-image-preview:
    rm -f build/credential-requester/preview/credential-requester.tar && \
    {{container}} save -o build/credential-requester/preview/credential-requester.tar credential-requester

# SCP the Credential Requester Docker image to the AWS parent instance (dev)
scp-credential-requester-to-aws-dev:
    scp -i $DEV_SSH_KEY build/credential-requester/dev/credential-requester.tar $DEV_SERVER:~/

# SCP the Docker image to the AWS parent instance (prod)
scp-credential-requester-to-aws-prod:
    scp -i $PROD_SSH_KEY build/credential-requester/prod/credential-requester.tar $PROD_SERVER:~/

# SCP the Credential Requester Docker image to the AWS parent instance (preview)
scp-credential-requester-to-aws-preview:
    scp -i $PREVIEW_SSH_KEY build/credential-requester/preview/credential-requester.tar $PREVIEW_SERVER:~/

# Load Credential Requester Docker image on AWS instance (dev)
load-credential-requester-docker-on-aws-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "docker load -i credential-requester.tar && docker tag localhost/credential-requester:latest credential-requester:latest"

# Load Credential Requester Docker image on AWS instance (prod)
load-credential-requester-docker-on-aws-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "docker load -i credential-requester.tar && docker tag localhost/credential-requester:latest credential-requester:latest"

# Load Credential Requester Docker image on AWS instance (preview)
load-credential-requester-docker-on-aws-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "docker load -i credential-requester.tar && docker tag localhost/credential-requester:latest credential-requester:latest"

# Run Credential Requester Docker image on AWS instance (dev)
run-credential-requester-docker-on-aws-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "docker run -d --restart always --name credential-requester --device=/dev/vsock:/dev/vsock -v /var/run/vsock:/var/run/vsock --privileged -e PORT=8003 credential-requester:latest"

# Run Credential Requester Docker image on AWS instance (prod)
run-credential-requester-docker-on-aws-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "docker run -d --restart always --name credential-requester --device=/dev/vsock:/dev/vsock -v /var/run/vsock:/var/run/vsock --privileged -e PORT=8003 credential-requester:latest"

# Run Credential Requester Docker image on AWS instance (preview)
run-credential-requester-docker-on-aws-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "docker run -d --restart always --name credential-requester --device=/dev/vsock:/dev/vsock -v /var/run/vsock:/var/run/vsock --privileged -e PORT=8003 credential-requester:latest"

### Logging Commands ###

# Build the Logging Docker image
build-logging-docker:
    {{container}} rmi enclave-logging:latest || true
    cd nitro-toolkit/logging && {{container}} build -t enclave-logging .

# Save Logging Docker image to a tar file (Dev)
save-logging-docker-image-dev:
    rm -f build/dev/logging/enclave-logging.tar && {{container}} save -o build/dev/logging/enclave-logging.tar enclave-logging

# Save Logging Docker image to a tar file (Prod)
save-logging-docker-image-prod:
    rm -f build/prod/logging/enclave-logging.tar && {{container}} save -o build/prod/logging/enclave-logging.tar enclave-logging

# Save Logging Docker image to a tar file (Preview)
save-logging-docker-image-preview:
    rm -f build/preview/logging/enclave-logging.tar && {{container}} save -o build/preview/logging/enclave-logging.tar enclave-logging

# SCP the Logging Docker image to the AWS parent instance (dev)
scp-logging-to-aws-dev:
    scp -i $DEV_SSH_KEY build/dev/logging/enclave-logging.tar $DEV_SERVER:~/

# SCP the Logging Docker image to the AWS parent instance (prod)
scp-logging-to-aws-prod:
    scp -i $PROD_SSH_KEY build/prod/logging/enclave-logging.tar $PROD_SERVER:~/

# SCP the Logging Docker image to the AWS parent instance (preview)
scp-logging-to-aws-preview:
    scp -i $PREVIEW_SSH_KEY build/preview/logging/enclave-logging.tar $PREVIEW_SERVER:~/

# Load Logging Docker image on AWS instance (dev)
load-logging-docker-on-aws-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "docker load -i enclave-logging.tar && docker tag localhost/enclave-logging:latest enclave-logging:latest"

# Load Logging Docker image on AWS instance (prod)
load-logging-docker-on-aws-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "docker load -i enclave-logging.tar && docker tag localhost/enclave-logging:latest enclave-logging:latest"

# Load Logging Docker image on AWS instance (preview)
load-logging-docker-on-aws-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "docker load -i enclave-logging.tar && docker tag localhost/enclave-logging:latest enclave-logging:latest"

# Run Logging Docker image on AWS instance (dev)
run-logging-docker-on-aws-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "docker run -d --restart always --name enclave-logging --device=/dev/vsock:/dev/vsock -v /var/run/vsock:/var/run/vsock --privileged -e VSOCK_PORT=8011 -e LOG_GROUP=/aws/nitro-enclaves/maple-enclave-dev -e LOG_STREAM=enclave-logs-dev -e AWS_REGION=us-east-2 enclave-logging:latest"

# Run Logging Docker image on AWS instance (prod)
run-logging-docker-on-aws-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "docker run -d --restart always --name enclave-logging --device=/dev/vsock:/dev/vsock -v /var/run/vsock:/var/run/vsock --privileged -e VSOCK_PORT=8011 -e LOG_GROUP=/aws/nitro-enclaves/maple-enclave-prod -e LOG_STREAM=enclave-logs-prod -e AWS_REGION=us-east-2 enclave-logging:latest"

# Run Logging Docker image on AWS instance (preview)
run-logging-docker-on-aws-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "docker run -d --restart always --name enclave-logging --device=/dev/vsock:/dev/vsock -v /var/run/vsock:/var/run/vsock --privileged -e VSOCK_PORT=8011 -e LOG_GROUP=/aws/nitro-enclaves/maple-enclave-preview -e LOG_STREAM=enclave-logs-preview -e AWS_REGION=us-east-2 enclave-logging:latest"

# Build and deploy logging for dev
build-and-deploy-logging-dev: build-logging-docker save-logging-docker-image-dev scp-logging-to-aws-dev load-logging-docker-on-aws-dev run-logging-docker-on-aws-dev

# Build and deploy logging for prod
build-and-deploy-logging-prod: build-logging-docker save-logging-docker-image-prod scp-logging-to-aws-prod load-logging-docker-on-aws-prod run-logging-docker-on-aws-prod

# Build and deploy logging for preview
build-and-deploy-logging-preview: build-logging-docker save-logging-docker-image-preview scp-logging-to-aws-preview load-logging-docker-on-aws-preview run-logging-docker-on-aws-preview

### Database Commands ###

# Setup diesel CLI (first-time setup)
diesel-setup:
    diesel setup

# Generate a new migration
diesel-migration-generate name:
    diesel migration generate {{name}}

# Run migrations locally
diesel-migration-run-local:
    diesel migration run

# Run migrations on development
diesel-migration-run-dev:
    diesel migration run --database-url $DEV_DATABASE_URL

# Run migrations on production
diesel-migration-run-prod:
    diesel migration run --database-url $PROD_DATABASE_URL

# Run migrations on preview
diesel-migration-run-preview:
    diesel migration run --database-url $PREVIEW_DATABASE_URL


### Continuum Proxy Commands ###

# Update continuum-proxy submodule to a specific version
update-continuum-proxy-version version:
    cd privatemode-public && git fetch --tags && git checkout {{version}}

# Build continuum-proxy from source using Nix (produces statically linked binary)
build-continuum-proxy:
    nix build ./privatemode-public#privatemode-proxy.bin -o continuum-proxy-build
    chmod u+w continuum-proxy || true
    cp continuum-proxy-build/bin/privatemode-proxy continuum-proxy
    chmod +x continuum-proxy
    rm continuum-proxy-build
    @echo "Built continuum-proxy:"
    @file continuum-proxy
    @./continuum-proxy --version

# Update continuum-proxy to a specific version and rebuild
update-continuum-proxy version="v1.39.1":
    just update-continuum-proxy-version {{version}}
    just build-continuum-proxy

### Local macOS Proxy Commands ###

# Build macOS-native Continuum and Tinfoil proxy binaries under .local/bin.
# Run from a Nix dev shell, for example: nix develop -c just build-local-proxies-macos
build-local-proxies-macos: build-continuum-proxy-macos build-tinfoil-proxy-macos

# Build a macOS-native Continuum proxy without replacing the checked-in Linux binary.
build-continuum-proxy-macos:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p .local/bin
    version="$(sed -n 's/.*version = "\([^"]*\)".*/\1/p' privatemode-public/version.nix)"
    if [ -z "$version" ]; then
        echo "Could not read Continuum version from privatemode-public/version.nix" >&2
        exit 1
    fi
    cd privatemode-public
    CGO_ENABLED=0 go build \
        -tags contrast_unstable_api \
        -ldflags "-X github.com/edgelesssys/continuum/internal/oss/constants.version=$version" \
        -o ../.local/bin/continuum-proxy-darwin \
        ./privatemode-proxy
    ../.local/bin/continuum-proxy-darwin --version

# Build a macOS-native Tinfoil proxy without replacing the checked-in Linux binary.
build-tinfoil-proxy-macos:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p .local/bin
    cd tinfoil-proxy
    CGO_ENABLED=0 go build -o ../.local/bin/tinfoil-proxy-darwin .
    file ../.local/bin/tinfoil-proxy-darwin

# Run the macOS-native Continuum proxy on CONTINUUM_PROXY_PORT, default 8092.
# The API key is read from CONTINUUM_API_KEY or .local/secrets/continuum_api_key.
run-continuum-proxy-macos:
    #!/usr/bin/env bash
    set -euo pipefail
    bin=".local/bin/continuum-proxy-darwin"
    key_file=".local/secrets/continuum_api_key"
    port="${CONTINUUM_PROXY_PORT:-8092}"
    workspace="${CONTINUUM_PROXY_WORKSPACE:-.local/continuum}"
    if [ ! -x "$bin" ]; then
        echo "$bin is missing. Run: nix develop -c just build-continuum-proxy-macos" >&2
        exit 1
    fi
    api_key="${CONTINUUM_API_KEY:-}"
    if [ -z "$api_key" ] && [ -f "$key_file" ]; then
        api_key="$(tr -d '\r\n' < "$key_file")"
    fi
    if [ -z "$api_key" ]; then
        echo "Set CONTINUUM_API_KEY or write the key to $key_file" >&2
        exit 1
    fi
    mkdir -p "$workspace"
    exec "$bin" --port "$port" --workspace "$workspace" --apiKey "$api_key"

# Run the macOS-native Tinfoil proxy on TINFOIL_PROXY_PORT, default 8093.
# The API key is read from TINFOIL_API_KEY or .local/secrets/tinfoil_api_key.
run-tinfoil-proxy-macos:
    #!/usr/bin/env bash
    set -euo pipefail
    bin=".local/bin/tinfoil-proxy-darwin"
    key_file=".local/secrets/tinfoil_api_key"
    port="${TINFOIL_PROXY_PORT:-8093}"
    if [ ! -x "$bin" ]; then
        echo "$bin is missing. Run: nix develop -c just build-tinfoil-proxy-macos" >&2
        exit 1
    fi
    api_key="${TINFOIL_API_KEY:-}"
    if [ -z "$api_key" ] && [ -f "$key_file" ]; then
        api_key="$(tr -d '\r\n' < "$key_file")"
    fi
    if [ -z "$api_key" ]; then
        echo "Set TINFOIL_API_KEY or write the key to $key_file" >&2
        exit 1
    fi
    TINFOIL_API_KEY="$api_key" TINFOIL_PROXY_PORT="$port" exec "$bin"

# Run the local OpenSecret backend pointed at the macOS proxy ports.
# Requires Postgres from nix develop and a populated .env.
run-local-backend-macos:
    APP_MODE="${APP_MODE:-local}" OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8092}" TINFOIL_API_BASE="${TINFOIL_API_BASE:-http://127.0.0.1:8093}" cargo run

### Enclave Management ###

# Terminate the running application enclave (dev)
# Skips p11ne (ACM/TLS enclave) - only terminates non-p11ne enclaves
# Does not fail if no enclave is running
terminate-enclave-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER 'bash -c "\
    ENCLAVE_ID=\$(nitro-cli describe-enclaves | jq -r \".[] | select(.EnclaveName != \\\"p11ne\\\") | .EnclaveID\" | head -1) && \
    if [ ! -z \"\$ENCLAVE_ID\" ] && [ \"\$ENCLAVE_ID\" != \"null\" ]; then \
        echo \"Terminating enclave with ID: \$ENCLAVE_ID\" && \
        nitro-cli terminate-enclave --enclave-id \$ENCLAVE_ID || true; \
    else \
        echo \"No application enclave running (p11ne is preserved).\"; \
    fi"'

# Terminate the running application enclave (prod)
# Skips p11ne (ACM/TLS enclave) - only terminates non-p11ne enclaves
# Does not fail if no enclave is running
terminate-enclave-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER 'bash -c "\
    ENCLAVE_ID=\$(nitro-cli describe-enclaves | jq -r \".[] | select(.EnclaveName != \\\"p11ne\\\") | .EnclaveID\" | head -1) && \
    if [ ! -z \"\$ENCLAVE_ID\" ] && [ \"\$ENCLAVE_ID\" != \"null\" ]; then \
        echo \"Terminating enclave with ID: \$ENCLAVE_ID\" && \
        nitro-cli terminate-enclave --enclave-id \$ENCLAVE_ID || true; \
    else \
        echo \"No application enclave running (p11ne is preserved).\"; \
    fi"'

# Terminate the running application enclave (preview)
# Skips p11ne (ACM/TLS enclave) - only terminates non-p11ne enclaves
# Does not fail if no enclave is running
terminate-enclave-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER 'bash -c "\
    ENCLAVE_ID=\$(nitro-cli describe-enclaves | jq -r \".[] | select(.EnclaveName != \\\"p11ne\\\") | .EnclaveID\" | head -1) && \
    if [ ! -z \"\$ENCLAVE_ID\" ] && [ \"\$ENCLAVE_ID\" != \"null\" ]; then \
        echo \"Terminating enclave with ID: \$ENCLAVE_ID\" && \
        nitro-cli terminate-enclave --enclave-id \$ENCLAVE_ID || true; \
    else \
        echo \"No application enclave running (p11ne is preserved).\"; \
    fi"'

# Restart socat-proxy service (dev)
restart-socat-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "sudo systemctl restart socat-proxy.service"

# Restart socat-proxy service (prod)
restart-socat-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "sudo systemctl restart socat-proxy.service"
#
# Restart socat-proxy service (preview)
restart-socat-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "sudo systemctl restart socat-proxy.service"

# Run the staged dev environment
run-stage-dev: terminate-enclave-dev run-eif-dev restart-socat-dev

# Run the staged prod environment
run-stage-prod: terminate-enclave-prod run-eif-prod restart-socat-prod

# Run the staged preview environment
run-stage-preview: terminate-enclave-preview run-eif-preview restart-socat-preview

### EIF Building ###

# Build the EIF using Nix
build-eif:
    nix build .?submodules=1#eif
    echo "EIF build completed. PCR:"
    cat result/pcr.json

# Build EIF for development environment
build-eif-dev:
    nix build .?submodules=1#eif-dev
    echo "EIF build completed. PCR:"
    cat result/pcr.json

# Build steady-state dev EIF for AEAD rollout with a distinct output path
build-eif-dev-steady:
    rm -f result-dev-steady
    nix build '.?submodules=1#eif-dev' -o result-dev-steady
    echo "Steady-state dev EIF build completed. PCR:"
    cat result-dev-steady/pcr.json

# Build EIF for production environment
build-eif-prod:
    nix build .?submodules=1#eif-prod
    echo "EIF build completed. PCR:"
    cat result/pcr.json

# Build steady-state production EIF for AEAD rollout with a distinct output path
build-eif-prod-steady:
    rm -f result-steady
    nix build '.?submodules=1#eif-prod' -o result-steady
    echo "Steady-state production EIF build completed. PCR:"
    cat result-steady/pcr.json

# Build one-off seed-wrap translation EIF for production AEAD rollout
build-eif-prod-seed-wrap-translation:
    rm -f result-translation
    nix build '.?submodules=1#eif-prod-seed-wrap-translation' -o result-translation
    echo "Seed-wrap translation production EIF build completed. PCR:"
    cat result-translation/pcr.json

# Build one-off seed-wrap translation EIF for development rehearsal
build-eif-dev-seed-wrap-translation:
    rm -f result-dev-translation
    nix build '.?submodules=1#eif-dev-seed-wrap-translation' -o result-dev-translation
    echo "Seed-wrap translation dev EIF build completed. PCR:"
    cat result-dev-translation/pcr.json

# Build EIF for preview environment
build-eif-preview:
    nix build .?submodules=1#eif-preview
    echo "EIF build completed. PCR:"
    cat result/pcr.json

# Build one-off seed-wrap translation EIF for preview rehearsal
build-eif-preview-seed-wrap-translation:
    rm -f result-preview-translation
    nix build '.?submodules=1#eif-preview-seed-wrap-translation' -o result-preview-translation
    echo "Seed-wrap translation preview EIF build completed. PCR:"
    cat result-preview-translation/pcr.json

# Build both production AEAD rollout artifacts without reusing the same result symlink
build-eif-prod-aead-rollout: build-eif-prod-seed-wrap-translation build-eif-prod-steady

# Build both dev AEAD rollout artifacts without reusing the same result symlink
build-eif-dev-aead-rollout: build-eif-dev-seed-wrap-translation build-eif-dev-steady

# Build EIF with custom environment variables
build-eif-custom env_vars:
    #!/usr/bin/env bash
    eval "{{env_vars}}" nix build .?submodules=1#eif
    echo "EIF build completed. PCR:"
    cat result/pcr.json

# Build EIF for development environment
copy-pcr-dev:
    nix build .?submodules=1#eif-dev
    echo "EIF build completed. PCR:"
    cat result/pcr.json
    cp -f result/pcr.json ./pcrDev.json

# Capture steady-state dev PCR for AEAD rollout
copy-pcr-dev-steady: build-eif-dev-steady
    cp -f result-dev-steady/pcr.json ./pcrDev.json

# Capture one-off seed-wrap translation dev PCR for AEAD rollout
copy-pcr-dev-seed-wrap-translation: build-eif-dev-seed-wrap-translation
    cp -f result-dev-translation/pcr.json ./pcrDevSeedWrapTranslation.json

# Build EIF for production environment
copy-pcr-prod:
    nix build .?submodules=1#eif-prod
    echo "EIF build completed. PCR:"
    cat result/pcr.json
    cp -f result/pcr.json ./pcrProd.json

# Capture steady-state production PCR for AEAD rollout
copy-pcr-prod-steady: build-eif-prod-steady
    cp -f result-steady/pcr.json ./pcrProd.json

# Capture one-off seed-wrap translation production PCR for AEAD rollout
copy-pcr-prod-seed-wrap-translation: build-eif-prod-seed-wrap-translation
    cp -f result-translation/pcr.json ./pcrProdSeedWrapTranslation.json

# Sign and append PCR measurements for dev environment
append-pcr-dev:
    #!/usr/bin/env bash
    set -e
    
    # Check for required environment variable
    if [ -z "${SIGNING_PRIVATE_KEY}" ]; then
        echo "❌ Error: SIGNING_PRIVATE_KEY environment variable is not set"
        echo "Please generate keys with: ./pcr_sign.js generate-keys"
        echo "Then set the environment variables with:"
        echo "export SIGNING_PRIVATE_KEY='...'"
        echo "export SIGNING_PUBLIC_KEY='...'"
        exit 1
    fi
    
    # Check if the Node.js script exists and is executable
    if [ ! -x "./pcr_sign.js" ]; then
        chmod +x ./pcr_sign.js
    fi
    
    # Initialize empty history file if it doesn't exist
    if [ ! -f "./pcrDevHistory.json" ]; then
        echo "[]" > ./pcrDevHistory.json
    fi
    
    # Get current PCR values
    PCR_CONTENT=$(cat ./pcrDev.json)
    CURRENT_PCR0=$(echo $PCR_CONTENT | jq -r '.PCR0')
    CURRENT_PCR1=$(echo $PCR_CONTENT | jq -r '.PCR1')
    CURRENT_PCR2=$(echo $PCR_CONTENT | jq -r '.PCR2')
    
    # Check if this PCR0 already exists in the history
    HISTORY=$(cat ./pcrDevHistory.json)
    PCR0_EXISTS=$(echo $HISTORY | jq --arg pcr0 "$CURRENT_PCR0" 'map(select(.PCR0 == $pcr0)) | length')
    
    if [ "$PCR0_EXISTS" -gt "0" ]; then
        echo "⚠️  PCR0 value already exists in pcrDevHistory.json"
        echo "    Skipping append operation to avoid duplicates."
        exit 0
    fi
    
    # Generate a timestamp
    TIMESTAMP=$(date +%s)
    
    # Sign just the PCR0 value
    echo "Signing PCR0: $CURRENT_PCR0"
    SIGNATURE=$(./pcr_sign.js sign-pcr0 "$CURRENT_PCR0")
    
    if [ -z "$SIGNATURE" ]; then
        echo "❌ Error: Failed to create signature"
        exit 1
    fi
    
    # Create a new history entry
    NEW_ENTRY=$(jq -n \
      --arg pcr0 "$CURRENT_PCR0" \
      --arg pcr1 "$CURRENT_PCR1" \
      --arg pcr2 "$CURRENT_PCR2" \
      --arg sig "$SIGNATURE" \
      --arg ts "$TIMESTAMP" \
      '{
        "PCR0": $pcr0, 
        "PCR1": $pcr1, 
        "PCR2": $pcr2, 
        "timestamp": ($ts | tonumber),
        "signature": $sig
      }')
    
    # Append to history file
    echo $HISTORY | jq --argjson entry "$NEW_ENTRY" '. + [$entry]' > ./pcrDevHistory.json
    
    echo "✅ Successfully appended signed PCR entry to pcrDevHistory.json"
    echo "   PCR0: $CURRENT_PCR0"
    echo "   Timestamp: $TIMESTAMP"

# Sign and append PCR measurements for prod environment
append-pcr-prod:
    #!/usr/bin/env bash
    set -e
    
    # Check for required environment variable
    if [ -z "${SIGNING_PRIVATE_KEY}" ]; then
        echo "❌ Error: SIGNING_PRIVATE_KEY environment variable is not set"
        echo "Please generate keys with: ./pcr_sign.js generate-keys"
        echo "Then set the environment variables with:"
        echo "export SIGNING_PRIVATE_KEY='...'"
        echo "export SIGNING_PUBLIC_KEY='...'"
        exit 1
    fi
    
    # Check if the Node.js script exists and is executable
    if [ ! -x "./pcr_sign.js" ]; then
        chmod +x ./pcr_sign.js
    fi
    
    # Initialize empty history file if it doesn't exist
    if [ ! -f "./pcrProdHistory.json" ]; then
        echo "[]" > ./pcrProdHistory.json
    fi
    
    # Get current PCR values
    PCR_CONTENT=$(cat ./pcrProd.json)
    CURRENT_PCR0=$(echo $PCR_CONTENT | jq -r '.PCR0')
    CURRENT_PCR1=$(echo $PCR_CONTENT | jq -r '.PCR1')
    CURRENT_PCR2=$(echo $PCR_CONTENT | jq -r '.PCR2')
    
    # Check if this PCR0 already exists in the history
    HISTORY=$(cat ./pcrProdHistory.json)
    PCR0_EXISTS=$(echo $HISTORY | jq --arg pcr0 "$CURRENT_PCR0" 'map(select(.PCR0 == $pcr0)) | length')
    
    if [ "$PCR0_EXISTS" -gt "0" ]; then
        echo "⚠️  PCR0 value already exists in pcrProdHistory.json"
        echo "    Skipping append operation to avoid duplicates."
        exit 0
    fi
    
    # Generate a timestamp
    TIMESTAMP=$(date +%s)
    
    # Sign just the PCR0 value
    echo "Signing PCR0: $CURRENT_PCR0"
    SIGNATURE=$(./pcr_sign.js sign-pcr0 "$CURRENT_PCR0")
    
    if [ -z "$SIGNATURE" ]; then
        echo "❌ Error: Failed to create signature"
        exit 1
    fi
    
    # Create a new history entry
    NEW_ENTRY=$(jq -n \
      --arg pcr0 "$CURRENT_PCR0" \
      --arg pcr1 "$CURRENT_PCR1" \
      --arg pcr2 "$CURRENT_PCR2" \
      --arg sig "$SIGNATURE" \
      --arg ts "$TIMESTAMP" \
      '{
        "PCR0": $pcr0, 
        "PCR1": $pcr1, 
        "PCR2": $pcr2, 
        "timestamp": ($ts | tonumber),
        "signature": $sig
      }')
    
    # Append to history file
    echo $HISTORY | jq --argjson entry "$NEW_ENTRY" '. + [$entry]' > ./pcrProdHistory.json
    
    echo "✅ Successfully appended signed PCR entry to pcrProdHistory.json"
    echo "   PCR0: $CURRENT_PCR0"
    echo "   Timestamp: $TIMESTAMP"

# Sign and append PCR measurements from a specific PCR file into a history file.
_append-pcr-file pcr_file history_file label:
    #!/usr/bin/env bash
    set -e

    PCR_FILE="./{{pcr_file}}"
    HISTORY_FILE="./{{history_file}}"

    # Check for required environment variable
    if [ -z "${SIGNING_PRIVATE_KEY}" ]; then
        echo "❌ Error: SIGNING_PRIVATE_KEY environment variable is not set"
        echo "Please generate keys with: ./pcr_sign.js generate-keys"
        echo "Then set the environment variables with:"
        echo "export SIGNING_PRIVATE_KEY='...'"
        echo "export SIGNING_PUBLIC_KEY='...'"
        exit 1
    fi

    # Check if the Node.js script exists and is executable
    if [ ! -x "./pcr_sign.js" ]; then
        chmod +x ./pcr_sign.js
    fi

    if [ ! -f "$PCR_FILE" ]; then
        echo "❌ Error: $PCR_FILE does not exist"
        exit 1
    fi

    # Initialize empty history file if it doesn't exist
    if [ ! -f "$HISTORY_FILE" ]; then
        echo "[]" > "$HISTORY_FILE"
    fi

    # Get current PCR values
    PCR_CONTENT=$(cat "$PCR_FILE")
    CURRENT_PCR0=$(echo "$PCR_CONTENT" | jq -r '.PCR0')
    CURRENT_PCR1=$(echo "$PCR_CONTENT" | jq -r '.PCR1')
    CURRENT_PCR2=$(echo "$PCR_CONTENT" | jq -r '.PCR2')

    # Check if this PCR0 already exists in the history
    HISTORY=$(cat "$HISTORY_FILE")
    PCR0_EXISTS=$(echo "$HISTORY" | jq --arg pcr0 "$CURRENT_PCR0" 'map(select(.PCR0 == $pcr0)) | length')

    if [ "$PCR0_EXISTS" -gt "0" ]; then
        echo "⚠️  PCR0 value already exists in {{history_file}} for {{label}}"
        echo "    Skipping append operation to avoid duplicates."
        exit 0
    fi

    # Generate a timestamp
    TIMESTAMP=$(date +%s)

    # Sign just the PCR0 value
    echo "Signing {{label}} PCR0: $CURRENT_PCR0"
    SIGNATURE=$(./pcr_sign.js sign-pcr0 "$CURRENT_PCR0")

    if [ -z "$SIGNATURE" ]; then
        echo "❌ Error: Failed to create signature"
        exit 1
    fi

    # Create a new history entry. Keep the existing history schema stable.
    NEW_ENTRY=$(jq -n \
      --arg pcr0 "$CURRENT_PCR0" \
      --arg pcr1 "$CURRENT_PCR1" \
      --arg pcr2 "$CURRENT_PCR2" \
      --arg sig "$SIGNATURE" \
      --arg ts "$TIMESTAMP" \
      '{
        "PCR0": $pcr0,
        "PCR1": $pcr1,
        "PCR2": $pcr2,
        "timestamp": ($ts | tonumber),
        "signature": $sig
      }')

    # Append to history file
    echo "$HISTORY" | jq --argjson entry "$NEW_ENTRY" '. + [$entry]' > "$HISTORY_FILE"

    echo "✅ Successfully appended signed PCR entry to {{history_file}}"
    echo "   Label: {{label}}"
    echo "   PCR0: $CURRENT_PCR0"
    echo "   Timestamp: $TIMESTAMP"

# Sign and append one-off seed-wrap translation PCR measurements for dev.
append-pcr-dev-seed-wrap-translation:
    just _append-pcr-file pcrDevSeedWrapTranslation.json pcrDevHistory.json dev-seed-wrap-translation

# Sign and append one-off seed-wrap translation PCR measurements for prod.
append-pcr-prod-seed-wrap-translation:
    just _append-pcr-file pcrProdSeedWrapTranslation.json pcrProdHistory.json prod-seed-wrap-translation

# Update PCR dev with signature and append to history
update-pcr-dev:
    just copy-pcr-dev
    just append-pcr-dev
    echo "✅ PCR dev values updated and history appended"

# Update PCR prod with signature and append to history
update-pcr-prod:
    just copy-pcr-prod
    just append-pcr-prod
    echo "✅ PCR prod values updated and history appended"

# Update dev PCRs for the AEAD rollout steady-state and one-off translation EIFs.
update-pcr-dev-aead-rollout:
    just copy-pcr-dev-steady
    just append-pcr-dev
    just copy-pcr-dev-seed-wrap-translation
    just append-pcr-dev-seed-wrap-translation
    echo "✅ AEAD rollout dev PCR values updated for steady-state and translation EIFs"

# Update prod PCRs for the AEAD rollout steady-state and one-off translation EIFs.
update-pcr-prod-aead-rollout:
    just copy-pcr-prod-steady
    just append-pcr-prod
    just copy-pcr-prod-seed-wrap-translation
    just append-pcr-prod-seed-wrap-translation
    echo "✅ AEAD rollout prod PCR values updated for steady-state and translation EIFs"

# Update all PCR values for both dev and prod environments
update-pcr-all:
    just update-pcr-dev-aead-rollout
    just update-pcr-prod-aead-rollout
    echo "✅ All AEAD rollout PCR values updated and history appended for dev and prod"


# Generate a key pair for PCR signing and output to terminal (no files created)
generate-pcr-keys:
    #!/usr/bin/env bash
    set -e
    
    # Check if the Node.js script exists and is executable
    if [ ! -x "./pcr_sign.js" ]; then
        chmod +x ./pcr_sign.js
    fi
    
    # Generate the keys using the Node.js script
    ./pcr_sign.js generate-keys

# Verify signatures in a PCR history file using the SIGNING_PUBLIC_KEY environment variable
verify-pcr-history env:
    #!/usr/bin/env bash
    set -e
    
    # Check if the Node.js script exists and is executable
    if [ ! -x "./pcr_verify.js" ]; then
        chmod +x ./pcr_verify.js
    fi
    
    # Check for required environment variable
    if [ -z "${SIGNING_PUBLIC_KEY}" ]; then
        echo "❌ Error: SIGNING_PUBLIC_KEY environment variable is not set"
        echo "Please generate keys with: ./pcr_sign.js generate-keys"
        echo "Then set the environment variables with:"
        echo "export SIGNING_PRIVATE_KEY='...'"
        echo "export SIGNING_PUBLIC_KEY='...'"
        exit 1
    fi
    
    # Display the first few characters of the public key for debugging
    PUBLIC_KEY_PREFIX="${SIGNING_PUBLIC_KEY:0:20}..."
    echo "Verifying signatures using public key: $PUBLIC_KEY_PREFIX"
    
    # Run the verification script
    ./pcr_verify.js {{env}}

# Internal function for PCR verification
_verify-pcr-internal env pcr_file:
    #!/usr/bin/env bash
    if [ ! -f "./{{pcr_file}}" ]; then
        echo "No {{pcr_file}} found. Building {{env}} EIF first..."
        just build-eif-{{env}}
        exit 0
    fi
    
    if [ ! -f result/pcr.json ]; then
        echo "No result/pcr.json found. Building {{env}} EIF first..."
        just build-eif-{{env}}
    fi
    
    if diff -q "./{{pcr_file}}" result/pcr.json > /dev/null; then
        echo "✅ {{env}} PCR values match!"
    else
        echo "❌ {{env}} PCR values do not match!"
        echo "Expected (./{{pcr_file}}):"
        cat "./{{pcr_file}}"
        echo "Got (result/pcr.json):"
        cat result/pcr.json
        exit 1
    fi

# Verify PCR values for dev environment
verify-pcr-dev:
    just _verify-pcr-internal dev pcrDev.json

# Verify steady-state dev PCR for AEAD rollout
verify-pcr-dev-steady:
    #!/usr/bin/env bash
    if [ ! -f ./pcrDev.json ]; then
        echo "No pcrDev.json found. Capturing steady-state dev PCR first..."
        just copy-pcr-dev-steady
        exit 0
    fi

    if [ ! -f result-dev-steady/pcr.json ]; then
        echo "No result-dev-steady/pcr.json found. Building steady-state dev EIF first..."
        just build-eif-dev-steady
    fi

    if diff -q ./pcrDev.json result-dev-steady/pcr.json > /dev/null; then
        echo "✅ Steady-state dev PCR values match!"
    else
        echo "❌ Steady-state dev PCR values do not match!"
        echo "Expected (./pcrDev.json):"
        cat ./pcrDev.json
        echo "Got (result-dev-steady/pcr.json):"
        cat result-dev-steady/pcr.json
        exit 1
    fi

# Verify PCR values for prod environment
verify-pcr-prod:
    just _verify-pcr-internal prod pcrProd.json

# Verify one-off seed-wrap translation dev PCR for AEAD rollout
verify-pcr-dev-seed-wrap-translation:
    #!/usr/bin/env bash
    if [ ! -f ./pcrDevSeedWrapTranslation.json ]; then
        echo "No pcrDevSeedWrapTranslation.json found. Capturing translation dev PCR first..."
        just copy-pcr-dev-seed-wrap-translation
        exit 0
    fi

    if [ ! -f result-dev-translation/pcr.json ]; then
        echo "No result-dev-translation/pcr.json found. Building translation dev EIF first..."
        just build-eif-dev-seed-wrap-translation
    fi

    if diff -q ./pcrDevSeedWrapTranslation.json result-dev-translation/pcr.json > /dev/null; then
        echo "✅ Seed-wrap translation dev PCR values match!"
    else
        echo "❌ Seed-wrap translation dev PCR values do not match!"
        echo "Expected (./pcrDevSeedWrapTranslation.json):"
        cat ./pcrDevSeedWrapTranslation.json
        echo "Got (result-dev-translation/pcr.json):"
        cat result-dev-translation/pcr.json
        exit 1
    fi

# Verify steady-state production PCR for AEAD rollout
verify-pcr-prod-steady:
    #!/usr/bin/env bash
    if [ ! -f ./pcrProd.json ]; then
        echo "No pcrProd.json found. Capturing steady-state production PCR first..."
        just copy-pcr-prod-steady
        exit 0
    fi

    if [ ! -f result-steady/pcr.json ]; then
        echo "No result-steady/pcr.json found. Building steady-state production EIF first..."
        just build-eif-prod-steady
    fi

    if diff -q ./pcrProd.json result-steady/pcr.json > /dev/null; then
        echo "✅ Steady-state production PCR values match!"
    else
        echo "❌ Steady-state production PCR values do not match!"
        echo "Expected (./pcrProd.json):"
        cat ./pcrProd.json
        echo "Got (result-steady/pcr.json):"
        cat result-steady/pcr.json
        exit 1
    fi

# Verify one-off seed-wrap translation production PCR for AEAD rollout
verify-pcr-prod-seed-wrap-translation:
    #!/usr/bin/env bash
    if [ ! -f ./pcrProdSeedWrapTranslation.json ]; then
        echo "No pcrProdSeedWrapTranslation.json found. Capturing translation production PCR first..."
        just copy-pcr-prod-seed-wrap-translation
        exit 0
    fi

    if [ ! -f result-translation/pcr.json ]; then
        echo "No result-translation/pcr.json found. Building translation production EIF first..."
        just build-eif-prod-seed-wrap-translation
    fi

    if diff -q ./pcrProdSeedWrapTranslation.json result-translation/pcr.json > /dev/null; then
        echo "✅ Seed-wrap translation production PCR values match!"
    else
        echo "❌ Seed-wrap translation production PCR values do not match!"
        echo "Expected (./pcrProdSeedWrapTranslation.json):"
        cat ./pcrProdSeedWrapTranslation.json
        echo "Got (result-translation/pcr.json):"
        cat result-translation/pcr.json
        exit 1
    fi

# Verify PCR values for preview environment
verify-pcr-preview:
    just _verify-pcr-internal preview pcrPreview.json

# Verify PCR values for custom environment
verify-pcr-custom:
    #!/usr/bin/env bash
    if [ ! -f ./pcrCustom.json ]; then
        echo "No pcrCustom.json found. Please run build-eif-custom first"
        exit 1
    fi
    
    if [ ! -f result/pcr.json ]; then
        echo "No result/pcr.json found. Please rebuild with the same environment variables"
        exit 1
    fi
    
    if diff -q ./pcrCustom.json result/pcr.json > /dev/null; then
        echo "✅ Custom PCR values match!"
    else
        echo "❌ Custom PCR values do not match!"
        echo "Expected (./pcrCustom.json):"
        cat ./pcrCustom.json
        echo "Got (result/pcr.json):"
        cat result/pcr.json
        exit 1
    fi

# SCP the Nix-built EIF to AWS parent instance (dev)
scp-eif-to-aws-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "rm -f ~/opensecret.eif"
    scp -i $DEV_SSH_KEY result/image.eif $DEV_SERVER:~/opensecret.eif

# SCP the one-off AEAD seed-wrap translation EIF to AWS parent instance (dev)
scp-eif-to-aws-dev-seed-wrap-translation:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "rm -f ~/opensecret-aead-translation.eif"
    scp -i $DEV_SSH_KEY result-dev-translation/image.eif $DEV_SERVER:~/opensecret-aead-translation.eif

# SCP the AEAD steady-state EIF to AWS parent instance (dev)
scp-eif-to-aws-dev-steady:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "rm -f ~/opensecret-aead-steady.eif"
    scp -i $DEV_SSH_KEY result-dev-steady/image.eif $DEV_SERVER:~/opensecret-aead-steady.eif

# SCP the Nix-built EIF to AWS parent instance (prod)
scp-eif-to-aws-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "rm -f ~/opensecret.eif"
    scp -i $PROD_SSH_KEY result/image.eif $PROD_SERVER:~/opensecret.eif

# SCP the one-off AEAD seed-wrap translation EIF to AWS parent instance (prod)
scp-eif-to-aws-prod-seed-wrap-translation:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "rm -f ~/opensecret-aead-translation.eif"
    scp -i $PROD_SSH_KEY result-translation/image.eif $PROD_SERVER:~/opensecret-aead-translation.eif

# SCP the AEAD steady-state EIF to AWS parent instance (prod)
scp-eif-to-aws-prod-steady:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "rm -f ~/opensecret-aead-steady.eif"
    scp -i $PROD_SSH_KEY result-steady/image.eif $PROD_SERVER:~/opensecret-aead-steady.eif

# SCP the Nix-built EIF to AWS parent instance (preview)
scp-eif-to-aws-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "rm -f ~/opensecret.eif"
    scp -i $PREVIEW_SSH_KEY result/image.eif $PREVIEW_SERVER:~/opensecret.eif

# Stage to dev environment without debug mode (using Nix-built EIF)
stage-dev-nix: build-eif-dev scp-eif-to-aws-dev

# Stage the one-off AEAD translation EIF to dev without running it
stage-dev-aead-translation: build-eif-dev-seed-wrap-translation scp-eif-to-aws-dev-seed-wrap-translation

# Stage the AEAD steady-state EIF to dev without running it
stage-dev-aead-steady: build-eif-dev-steady scp-eif-to-aws-dev-steady

# Stage to prod environment without debug mode (using Nix-built EIF)
stage-prod-nix: build-eif-prod scp-eif-to-aws-prod

# Stage the one-off AEAD translation EIF to prod without running it
stage-prod-aead-translation: build-eif-prod-seed-wrap-translation scp-eif-to-aws-prod-seed-wrap-translation

# Stage the AEAD steady-state EIF to prod without running it
stage-prod-aead-steady: build-eif-prod-steady scp-eif-to-aws-prod-steady

# Stage to preview environment without debug mode (using Nix-built EIF)
stage-preview-nix: build-eif-preview scp-eif-to-aws-preview

# Run EIF file on AWS (dev)
run-eif-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4"

# Run the one-off AEAD seed-wrap translation EIF on AWS (dev)
run-eif-dev-seed-wrap-translation:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "nitro-cli run-enclave --eif-path opensecret-aead-translation.eif --memory 16384 --cpu-count 4"

# Run the AEAD steady-state EIF on AWS (dev)
run-eif-dev-steady:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "nitro-cli run-enclave --eif-path opensecret-aead-steady.eif --memory 16384 --cpu-count 4"

# Run EIF file on AWS (prod)
run-eif-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4"

# Run the one-off AEAD seed-wrap translation EIF on AWS (prod)
run-eif-prod-seed-wrap-translation:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "nitro-cli run-enclave --eif-path opensecret-aead-translation.eif --memory 16384 --cpu-count 4"

# Run the AEAD steady-state EIF on AWS (prod)
run-eif-prod-steady:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "nitro-cli run-enclave --eif-path opensecret-aead-steady.eif --memory 16384 --cpu-count 4"

# Run EIF file on AWS (preview)
run-eif-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4"

# Run EIF file in debug mode (preview)
run-eif-debug-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4 --debug-mode"

# Run EIF file in debug mode (dev)
run-eif-debug-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4 --debug-mode"

# Run EIF file in debug mode (prod)
run-eif-debug-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4 --debug-mode"

# View console logs in debug mode (dev)
view-console-logs-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "export ENCLAVE_ID=$(nitro-cli describe-enclaves | jq -r '.[0].EnclaveID') && nitro-cli console --enclave-id $ENCLAVE_ID"

# View console logs in debug mode (prod)
view-console-logs-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "export ENCLAVE_ID=$(nitro-cli describe-enclaves | jq -r '.[0].EnclaveID') && nitro-cli console --enclave-id $ENCLAVE_ID"

# SSH into prod server with a custom command
ssh-prod CMD:
    ssh -i $PROD_SSH_KEY $PROD_SERVER {{quote(CMD)}}

# View console logs in debug mode (preview)
view-console-logs-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "export ENCLAVE_ID=$(nitro-cli describe-enclaves | jq -r '.[0].EnclaveID') && nitro-cli console --enclave-id $ENCLAVE_ID"

# Deploy to dev environment without debug mode (using Nix-built EIF)
deploy-dev-nix: build-eif-dev verify-pcr-dev scp-eif-to-aws-dev
    @echo "EIF copied to server. Please review the PCR values and press Enter to continue with termination and deployment..."
    @read -p ""
    just terminate-enclave-dev run-eif-dev restart-socat-dev

# Deploy to prod environment without debug mode (using Nix-built EIF)
deploy-prod-nix: build-eif-prod verify-pcr-prod scp-eif-to-aws-prod
    @echo "EIF copied to production server. Please review the PCR values and press Enter to continue with termination and deployment..."
    @read -p ""
    just terminate-enclave-prod run-eif-prod restart-socat-prod

# Deploy to preview environment without debug mode (using Nix-built EIF)
deploy-preview-nix: build-eif-preview verify-pcr-preview scp-eif-to-aws-preview
    @echo "EIF copied to preview server. Please review the PCR values and press Enter to continue with termination and deployment..."
    @read -p ""
    just terminate-enclave-preview run-eif-preview restart-socat-preview

# Clean EIF build artifacts
clean-eif:
    rm -f result result-dev-steady result-dev-translation result-steady result-translation result-preview-translation

### Tinfoil Proxy Commands ###

# Build tinfoil-proxy binary using Go
build-tinfoil-proxy:
    {{container}} build --platform linux/arm64 -t tinfoil-proxy-builder -f tinfoil-proxy/Dockerfile tinfoil-proxy
    {{container}} create --name tinfoil-proxy-extract tinfoil-proxy-builder
    mkdir -p tinfoil-proxy/dist
    {{container}} cp tinfoil-proxy-extract:/tinfoil-proxy tinfoil-proxy/dist/
    {{container}} rm tinfoil-proxy-extract
    echo "Go binary created at: tinfoil-proxy/dist/tinfoil-proxy"
    echo "Size: $(du -h tinfoil-proxy/dist/tinfoil-proxy | cut -f1)"
    file tinfoil-proxy/dist/tinfoil-proxy

# Clean tinfoil-proxy build artifacts
clean-tinfoil-proxy:
    rm -rf tinfoil-proxy/dist
