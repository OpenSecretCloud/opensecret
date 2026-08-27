# Load environment variables from .env file
set dotenv-load

# Set the container runtime (docker or podman)
container := "podman"

# Set the default recipe to list all available commands
default:
    @just --list

# Build the pinned NSM and KMS helper sources with Nix
build-nitro-bins:
    nix build --no-update-lock-file .#nitro-bins

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
    nix build --no-update-lock-file ./privatemode-public#privatemode-proxy.bin -o continuum-proxy-build
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

# Build the macOS-native Continuum proxy binary under .local/bin.
# Run from a Nix dev shell, for example: nix develop --no-update-lock-file -c just build-local-proxies-macos
build-local-proxies-macos: build-continuum-proxy-macos

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
        echo "$bin is missing. Run: nix develop --no-update-lock-file -c just build-continuum-proxy-macos" >&2
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
    exec "$bin" --port "$port" --workspace "$workspace" --apiKey "$api_key" --sharedPromptCache

# Run the local OpenSecret backend with Continuum's local proxy and the
# in-process Tinfoil SDK. Requires Postgres and a populated .env.
run-local-backend-macos:
    #!/usr/bin/env bash
    set -euo pipefail
    key_file=".local/secrets/tinfoil_api_key"
    tinfoil_api_key="${TINFOIL_API_KEY:-}"
    if [ -z "$tinfoil_api_key" ] && [ -f "$key_file" ]; then
        tinfoil_api_key="$(tr -d '\r\n' < "$key_file")"
    fi
    if [ -z "$tinfoil_api_key" ]; then
        echo "Set TINFOIL_API_KEY or write the key to $key_file" >&2
        exit 1
    fi
    APP_MODE="${APP_MODE:-local}" \
        OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8092}" \
        TINFOIL_API_KEY="$tinfoil_api_key" \
        exec cargo run --locked

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

# Split staging cannot bind the later run to one approved artifact.
run-stage-dev:
    @echo "❌ Use 'just deploy-dev-nix <vMAJOR.MINOR.PATCH>' for an atomic, verified release deployment." >&2
    @exit 1

# Split staging cannot bind the later run to one approved artifact.
run-stage-prod:
    @echo "❌ Use 'just deploy-prod-nix <vMAJOR.MINOR.PATCH>' for an atomic, verified release deployment." >&2
    @exit 1

# Run the staged preview environment
run-stage-preview: terminate-enclave-preview run-eif-preview restart-socat-preview

### EIF Building ###

# Build EIF for development environment
build-eif-dev:
    nix build --no-update-lock-file '.?submodules=1#eif-dev'
    echo "EIF build completed. PCR:"
    cat result/pcr.json

# Build EIF for production environment
build-eif-prod:
    nix build --no-update-lock-file '.?submodules=1#eif-prod'
    echo "EIF build completed. PCR:"
    cat result/pcr.json

# Build EIF for preview environment
build-eif-preview:
    nix build --no-update-lock-file '.?submodules=1#eif-preview'
    echo "EIF build completed. PCR:"
    cat result/pcr.json

# Build EIF for development environment
copy-pcr-dev:
    nix build --no-update-lock-file '.?submodules=1#eif-dev'
    echo "EIF build completed. PCR:"
    cat result/pcr.json
    cp -f result/pcr.json ./pcrDev.json

# Build EIF for production environment
copy-pcr-prod:
    nix build --no-update-lock-file '.?submodules=1#eif-prod'
    echo "EIF build completed. PCR:"
    cat result/pcr.json
    cp -f result/pcr.json ./pcrProd.json

# Internal transition-only compatibility primitive. Release operators must use
# append-legacy-pcr-release so both Sigstore publications are authenticated
# before either old PCR0-only signature is created.
_append-pcr-file pcr_file history_file environment:
    #!/usr/bin/env bash
    set -euo pipefail

    pcr_file={{quote(pcr_file)}}
    history_file={{quote(history_file)}}

    if [[ ! -f "$pcr_file" ]]; then
        echo "❌ Required PCR file does not exist: $pcr_file" >&2
        exit 1
    fi
    if [[ ! -f "$history_file" ]]; then
        echo "❌ Refusing to create missing append-only history: $history_file" >&2
        exit 1
    fi

    history_git_path="${history_file#./}"
    base_history="$(mktemp "${TMPDIR:-/tmp}/opensecret-legacy-head.XXXXXX")"
    temporary_history=""
    cleanup() {
        if [[ -n "$temporary_history" ]]; then
            rm -f -- "$temporary_history"
        fi
        rm -f -- "$base_history"
    }
    trap cleanup EXIT

    if ! git show "HEAD:$history_git_path" > "$base_history"; then
        echo "❌ Could not load $history_git_path from HEAD." >&2
        exit 1
    fi
    if ! git diff --cached --quiet -- "$history_file" &&
       ! git diff --quiet -- "$history_file"; then
        echo "❌ Legacy history has different staged and unstaged changes." >&2
        exit 1
    fi

    # A retry may see the one valid suffix written by an earlier partial
    # update-pcr-all run. Validate the working history against HEAD before
    # inspecting it; truncation, reordering, mutation, bad signatures, and
    # cross-environment PCR0 reuse all fail here.
    ./pcr_verify.js \
        {{environment}} \
        --history-file "$history_file" \
        --base-history-file "$base_history"

    pcr0="$(jq -er '.PCR0' "$pcr_file")"
    pcr1="$(jq -er '.PCR1' "$pcr_file")"
    pcr2="$(jq -er '.PCR2' "$pcr_file")"
    exact_matches="$(
        jq \
            --arg pcr0 "$pcr0" \
            --arg pcr1 "$pcr1" \
            --arg pcr2 "$pcr2" \
            '[.[] | select(.PCR0 == $pcr0 and .PCR1 == $pcr1 and .PCR2 == $pcr2)] | length' \
            "$history_file"
    )"
    pcr0_matches="$(
        jq --arg pcr0 "$pcr0" '[.[] | select(.PCR0 == $pcr0)] | length' "$history_file"
    )"

    history_differs=0
    if ! git diff --quiet HEAD -- "$history_file"; then
        history_differs=1
    fi
    if [[ "$exact_matches" == "1" && "$pcr0_matches" == "1" ]]; then
        ./pcr_verify.js \
            {{environment}} \
            --history-file "$history_file" \
            --base-history-file "$base_history" \
            --require-pcr-file "$pcr_file"

        if [[ "$history_differs" == "1" ]]; then
            base_pcr0_matches="$(
                jq --arg pcr0 "$pcr0" \
                    '[.[] | select(.PCR0 == $pcr0)] | length' \
                    "$base_history"
            )"
            base_entries="$(jq -er 'length' "$base_history")"
            candidate_entries="$(jq -er 'length' "$history_file")"
            if [[ "$base_pcr0_matches" != "0" ]] ||
               (( candidate_entries != base_entries + 1 )); then
                echo "❌ Dirty history is not exactly the requested one-entry append." >&2
                exit 1
            fi
        fi

        echo "✅ {{environment}} legacy history already contains the exact PCR tuple."
        exit 0
    fi
    if [[ "$pcr0_matches" != "0" ]]; then
        echo "❌ {{environment}} history already contains PCR0 with a different or duplicate tuple." >&2
        exit 1
    fi
    if [[ "$history_differs" == "1" ]]; then
        echo "❌ {{environment}} history differs from HEAD but lacks the requested exact tuple." >&2
        exit 1
    fi
    if [[ -z "${SIGNING_PRIVATE_KEY:-}" ]]; then
        echo "❌ SIGNING_PRIVATE_KEY must contain the existing legacy PKCS#8 DER key encoded as base64." >&2
        exit 1
    fi

    signature="$(./pcr_sign.js sign-pcr0 "$pcr0")"
    timestamp="$(date +%s)"
    temporary_history="$(mktemp "./.${history_git_path##*/}.XXXXXX")"

    jq \
        --arg pcr0 "$pcr0" \
        --arg pcr1 "$pcr1" \
        --arg pcr2 "$pcr2" \
        --arg signature "$signature" \
        --argjson timestamp "$timestamp" \
        '. + [{
            PCR0: $pcr0,
            PCR1: $pcr1,
            PCR2: $pcr2,
            timestamp: $timestamp,
            signature: $signature
        }]' \
        "$history_file" > "$temporary_history"
    ./pcr_verify.js \
        {{environment}} \
        --history-file "$temporary_history" \
        --base-history-file "$base_history" \
        --require-pcr-file "$pcr_file"

    # Do not overwrite a concurrent operator's edit.
    if ! cmp -s "$base_history" "$history_file"; then
        echo "❌ Legacy history changed while the append was being prepared." >&2
        exit 1
    fi
    mv "$temporary_history" "$history_file"
    temporary_history=""

    ./pcr_verify.js \
        {{environment}} \
        --history-file "$history_file" \
        --base-history-file "$base_history" \
        --require-pcr-file "$pcr_file"
    echo "✅ Appended the {{environment}} tuple for legacy clients."

prepare-pcr-references:
    just copy-pcr-dev
    just copy-pcr-prod

update-pcr-prod:
    @echo "❌ Use 'just prepare-pcr-references' before tagging, then 'just update-pcr-all <tag>' after Sigstore publication." >&2
    @exit 1

update-pcr-dev:
    @echo "❌ Use 'just prepare-pcr-references' before tagging, then 'just update-pcr-all <tag>' after Sigstore publication." >&2
    @exit 1

update-pcr-all release_tag:
    just append-legacy-pcr-release {{quote(release_tag)}}

# Do not rotate this key: released clients pin the existing public key.
generate-pcr-keys:
    @echo "❌ Legacy key generation is disabled; recover the existing SIGNING_PRIVATE_KEY." >&2
    @exit 1

# Verify every history entry using the public key pinned by legacy clients.
verify-pcr-history env:
    ./pcr_verify.js {{env}}

# Require a local history to contain the exact complete tuple.
verify-legacy-pcr-compatibility env pcr_file:
    ./pcr_verify.js {{env}} --require-pcr-file {{quote(pcr_file)}}

# Require GitHub's master history—the URL used by old clients—to expose the
# exact tuple before terminating a running enclave.
verify-legacy-pcr-published env pcr_file:
    #!/usr/bin/env bash
    set -euo pipefail

    case "{{env}}" in
        dev)
            history_url="https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrDevHistory.json"
            other_history_url="https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrProdHistory.json"
            base_history="pcrDevHistory.json"
            other_base_history="pcrProdHistory.json"
            other_environment="prod"
            ;;
        prod)
            history_url="https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrProdHistory.json"
            other_history_url="https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrDevHistory.json"
            base_history="pcrProdHistory.json"
            other_base_history="pcrDevHistory.json"
            other_environment="dev"
            ;;
        *)
            echo "❌ Legacy publication exists only for dev or prod." >&2
            exit 1
            ;;
    esac
    test -s "$base_history"
    test -s "$other_base_history"

    published_history="$(mktemp "${TMPDIR:-/tmp}/opensecret-legacy-history.XXXXXX")"
    published_other_history="$(mktemp "${TMPDIR:-/tmp}/opensecret-legacy-other.XXXXXX")"
    trap 'rm -f "$published_history" "$published_other_history"' EXIT
    fetch_history() {
        local url="$1"
        local output="$2"
        curl \
            --proto '=https' \
            --tlsv1.2 \
            --connect-timeout 10 \
            --max-time 30 \
            --retry 3 \
            --retry-delay 2 \
            --fail \
            --silent \
            --show-error \
            --location \
            "$url" \
            --output "$output"
    }
    fetch_history "$history_url" "$published_history"
    fetch_history "$other_history_url" "$published_other_history"
    ./pcr_verify.js \
        {{env}} \
        --history-file "$published_history" \
        --other-history-file "$published_other_history" \
        --base-history-file "$base_history" \
        --require-pcr-file {{quote(pcr_file)}}
    ./pcr_verify.js \
        "$other_environment" \
        --history-file "$published_other_history" \
        --other-history-file "$published_history" \
        --base-history-file "$other_base_history"
    echo "✅ GitHub master exposes the {{env}} tuple as append-only dev/prod histories."

# Require two successful raw-GitHub reads separated by ten minutes. The
# endpoint currently advertises max-age=300; this conservative soak reduces
# the chance that an old client reaches a still-stale CDN edge.
verify-legacy-pcr-propagated env pcr_file:
    #!/usr/bin/env bash
    set -euo pipefail

    environment={{quote(env)}}
    pcr_file={{quote(pcr_file)}}
    case "$environment" in
        dev|prod) ;;
        *)
            echo "❌ Legacy publication exists only for dev or prod." >&2
            exit 1
            ;;
    esac
    test -s "$pcr_file"

    just verify-legacy-pcr-published "$environment" "$pcr_file"

    pcr0="$(jq -er '.PCR0' "$pcr_file")"
    marker_dir=".local/legacy-pcr-soak"
    marker="$marker_dir/${environment}-${pcr0}.timestamp"
    minimum_age=600
    now="$(date +%s)"
    mkdir -p "$marker_dir"

    if [[ ! -f "$marker" ]]; then
        printf '%s\n' "$now" > "$marker"
        echo "⏳ First verified raw-GitHub observation recorded for $environment." >&2
        echo "   Wait at least $minimum_age seconds, then rerun this command." >&2
        exit 1
    fi

    first_observation="$(tr -d '\r\n' < "$marker")"
    if [[ ! "$first_observation" =~ ^[0-9]+$ ]] ||
       (( first_observation > now )); then
        echo "❌ Invalid propagation marker: $marker" >&2
        exit 1
    fi
    age=$((now - first_observation))
    if (( age < minimum_age )); then
        echo "⏳ Raw-GitHub propagation soak has run for ${age}s; ${minimum_age}s required." >&2
        exit 1
    fi

    # The first read above is deliberately repeated after the elapsed-time
    # check so deployment never relies on an old successful observation.
    just verify-legacy-pcr-published "$environment" "$pcr_file"
    echo "✅ Two exact raw-GitHub observations are separated by at least ${minimum_age}s."

# Authenticate a published tagged manifest and bind it to the exact local EIF.
verify-sigstore-release-published release_tag environment pcr_file eif_file allow_legacy_history_changes='false':
    #!/usr/bin/env bash
    set -euo pipefail

    release_tag={{quote(release_tag)}}
    environment={{quote(environment)}}
    pcr_file={{quote(pcr_file)}}
    eif_file={{quote(eif_file)}}
    allow_legacy_history_changes={{quote(allow_legacy_history_changes)}}
    repository="OpenSecretCloud/opensecret"
    legacy_check_dir=""
    release_dir=""
    cleanup() {
        if [[ -n "$legacy_check_dir" ]]; then
            rm -rf -- "$legacy_check_dir"
        fi
        if [[ -n "$release_dir" ]]; then
            rm -rf -- "$release_dir"
        fi
    }
    trap cleanup EXIT

    if [[ ! "$release_tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
        echo "❌ Release tag must match exactly vMAJOR.MINOR.PATCH." >&2
        exit 1
    fi
    if [[ "$environment" != "dev" && "$environment" != "prod" ]]; then
        echo "❌ Release environment must be dev or prod." >&2
        exit 1
    fi
    if [[ "$allow_legacy_history_changes" != "false" &&
          "$allow_legacy_history_changes" != "true" ]]; then
        echo "❌ Internal legacy-history mode must be true or false." >&2
        exit 1
    fi
    for command in cosign gh git jq python3; do
        if ! command -v "$command" >/dev/null; then
            echo "❌ Required release verifier is missing: $command" >&2
            echo "   Enter the pinned Nix development shell first." >&2
            exit 1
        fi
    done
    if [[ ! -s "$pcr_file" || ! -s "$eif_file" ]]; then
        echo "❌ Required local PCR or EIF file is missing." >&2
        exit 1
    fi

    cosign_version="$(cosign version 2>&1)"
    if ! grep -Eq 'GitVersion:[[:space:]]+v?3\.1\.2([[:space:]]|$)' <<<"$cosign_version"; then
        echo "❌ Deployment verification requires exactly Cosign 3.1.2." >&2
        printf '%s\n' "$cosign_version" >&2
        exit 1
    fi

    tag_commit="$(git rev-parse --verify "${release_tag}^{commit}")"
    if [[ "$(git rev-parse HEAD)" != "$tag_commit" ]]; then
        echo "❌ Check out the exact tagged commit before deployment." >&2
        exit 1
    fi
    if [[ "$allow_legacy_history_changes" == "false" ]]; then
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo "❌ Refusing to verify deployment from a tracked worktree that differs from the tag." >&2
            exit 1
        fi
    else
        changed_files="$(
            {
                git diff --name-only
                git diff --cached --name-only
            } | LC_ALL=C sort -u
        )"
        while IFS= read -r changed_file; do
            case "$changed_file" in
                ""|pcrDevHistory.json|pcrProdHistory.json) ;;
                *)
                    echo "❌ Only append-only legacy histories may differ while preparing compatibility." >&2
                    exit 1
                    ;;
            esac
        done <<<"$changed_files"

        legacy_check_dir="$(mktemp -d "${TMPDIR:-/tmp}/opensecret-legacy-tag.XXXXXX")"
        git show "${tag_commit}:pcrDevHistory.json" > "$legacy_check_dir/pcrDevHistory.json"
        git show "${tag_commit}:pcrProdHistory.json" > "$legacy_check_dir/pcrProdHistory.json"
        ./pcr_verify.js dev \
            --base-history-file "$legacy_check_dir/pcrDevHistory.json"
        ./pcr_verify.js prod \
            --base-history-file "$legacy_check_dir/pcrProdHistory.json"
        rm -rf -- "$legacy_check_dir"
        legacy_check_dir=""
    fi

    remote_tag_commit="$(
        gh api "repos/$repository/commits/$release_tag" --jq .sha
    )"
    if [[ "$remote_tag_commit" != "$tag_commit" ]]; then
        echo "❌ GitHub's current tag does not resolve to the authenticated local commit." >&2
        exit 1
    fi

    release_state="$(
        gh release view "$release_tag" \
            --repo "$repository" \
            --json isDraft,isImmutable,isPrerelease,tagName \
            --jq '[.tagName, .isDraft, .isPrerelease, .isImmutable] | @tsv'
    )"
    expected_release_state="${release_tag}"$'\tfalse\tfalse\ttrue'
    if [[ "$release_state" != "$expected_release_state" ]]; then
        echo "❌ $release_tag is not a published, stable, immutable GitHub Release." >&2
        exit 1
    fi

    release_dir="$(mktemp -d "${TMPDIR:-/tmp}/opensecret-release.XXXXXX")"
    manifest_name="opensecret-nitro-${release_tag}-${environment}.manifest.json"
    bundle_name="opensecret-nitro-${release_tag}-${environment}.manifest.sigstore.json"
    gh release download "$release_tag" \
        --repo "$repository" \
        --pattern "$manifest_name" \
        --pattern "$bundle_name" \
        --dir "$release_dir"

    manifest="$release_dir/$manifest_name"
    bundle="$release_dir/$bundle_name"
    test -s "$manifest"
    test -s "$bundle"
    jq -e \
        '.mediaType == "application/vnd.dev.sigstore.bundle.v0.3+json"' \
        "$bundle" >/dev/null

    workflow_run="$(jq -er '.build.workflowRun' "$manifest")"
    python3 scripts/generate_nitro_release_manifest.py verify \
        --environment "$environment" \
        --commit "$tag_commit" \
        --tag "$release_tag" \
        --workflow-run "$workflow_run" \
        --pcr "$pcr_file" \
        --eif "$eif_file" \
        --flake-lock flake.lock \
        --manifest "$manifest"

    certificate_identity="https://github.com/${repository}/.github/workflows/release-nitro-eif.yml@refs/tags/${release_tag}"
    cosign verify-blob \
        --bundle "$bundle" \
        --certificate-github-workflow-name "Nitro EIF Release" \
        --certificate-github-workflow-ref "refs/tags/$release_tag" \
        --certificate-github-workflow-repository "$repository" \
        --certificate-github-workflow-sha "$tag_commit" \
        --certificate-github-workflow-trigger "workflow_dispatch" \
        --certificate-identity "$certificate_identity" \
        --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
        "$manifest"

    echo "✅ $release_tag authenticates the exact local $environment EIF and PCR tuple."

# After an immutable tagged release exists, authenticate both release outputs
# and only then create/reuse the legacy PCR0 entries needed by old clients.
append-legacy-pcr-release release_tag:
    #!/usr/bin/env bash
    set -euo pipefail

    release_tag={{quote(release_tag)}}
    if [[ ! "$release_tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
        echo "❌ Release tag must match exactly vMAJOR.MINOR.PATCH." >&2
        exit 1
    fi

    tag_commit="$(git rev-parse --verify "${release_tag}^{commit}")"
    if [[ "$(git rev-parse HEAD)" != "$tag_commit" ]]; then
        echo "❌ Check out the exact released tag before preparing legacy compatibility." >&2
        exit 1
    fi

    build_dir=".local/release-builds/$release_tag"
    install -d -m 0700 "$build_dir"
    nix build --no-update-lock-file \
        --out-link "$build_dir/dev" \
        '.?submodules=1#eif-dev'
    nix build --no-update-lock-file \
        --out-link "$build_dir/prod" \
        '.?submodules=1#eif-prod'

    dev_result="$(realpath "$build_dir/dev")"
    prod_result="$(realpath "$build_dir/prod")"
    for result_dir in "$dev_result" "$prod_result"; do
        if [[ "$result_dir" != /nix/store/* ]]; then
            echo "❌ Release build must resolve to an immutable Nix store output." >&2
            exit 1
        fi
        test -s "$result_dir/pcr.json"
        test -s "$result_dir/image.eif"
    done

    if ! cmp -s pcrDev.json "$dev_result/pcr.json" ||
       ! cmp -s pcrProd.json "$prod_result/pcr.json"; then
        echo "❌ Tagged builds do not match the checked-in PCR references." >&2
        exit 1
    fi

    # Both Sigstore releases are authenticated before either legacy history is
    # changed. The true mode permits only already-validated history suffixes so
    # an interrupted dev/prod append can be retried safely.
    just verify-sigstore-release-published \
        "$release_tag" dev \
        "$dev_result/pcr.json" "$dev_result/image.eif" true
    just verify-sigstore-release-published \
        "$release_tag" prod \
        "$prod_result/pcr.json" "$prod_result/image.eif" true

    just _append-pcr-file \
        "$dev_result/pcr.json" pcrDevHistory.json dev
    just _append-pcr-file \
        "$prod_result/pcr.json" pcrProdHistory.json prod

    echo "✅ Legacy compatibility now matches the authenticated $release_tag release."
    echo "   Commit only the suffix-only history changes and merge them to protected master."

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

# Verify PCR values for prod environment
verify-pcr-prod:
    just _verify-pcr-internal prod pcrProd.json

# Verify PCR values for preview environment
verify-pcr-preview:
    just _verify-pcr-internal preview pcrPreview.json

# SCP the Nix-built EIF to AWS parent instance (dev)
scp-eif-to-aws-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "rm -f ~/opensecret.eif"
    scp -i $DEV_SSH_KEY result/image.eif $DEV_SERVER:~/opensecret.eif

# SCP the Nix-built EIF to AWS parent instance (prod)
scp-eif-to-aws-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "rm -f ~/opensecret.eif"
    scp -i $PROD_SSH_KEY result/image.eif $PROD_SERVER:~/opensecret.eif

# SCP the Nix-built EIF to AWS parent instance (preview)
scp-eif-to-aws-preview:
    ssh -i $PREVIEW_SSH_KEY $PREVIEW_SERVER "rm -f ~/opensecret.eif"
    scp -i $PREVIEW_SSH_KEY result/image.eif $PREVIEW_SERVER:~/opensecret.eif

# Stage to dev environment without debug mode (using Nix-built EIF)
stage-dev-nix: build-eif-dev scp-eif-to-aws-dev

# Stage to prod environment without debug mode (using Nix-built EIF)
stage-prod-nix: build-eif-prod scp-eif-to-aws-prod

# Stage to preview environment without debug mode (using Nix-built EIF)
stage-preview-nix: build-eif-preview scp-eif-to-aws-preview

# Run EIF file on AWS (dev)
run-eif-dev:
    ssh -i $DEV_SSH_KEY $DEV_SERVER "nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4"

# Run EIF file on AWS (prod)
run-eif-prod:
    ssh -i $PROD_SSH_KEY $PROD_SERVER "nitro-cli run-enclave --eif-path opensecret.eif --memory 16384 --cpu-count 4"

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

# Deploy one exact tagged dev/prod EIF after both trust paths are published.
_deploy-tagged-nitro-eif release_tag environment:
    #!/usr/bin/env bash
    set -euo pipefail

    release_tag={{quote(release_tag)}}
    environment={{quote(environment)}}
    if [[ ! "$release_tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
        echo "❌ Release tag must match exactly vMAJOR.MINOR.PATCH." >&2
        exit 1
    fi

    case "$environment" in
        dev)
            ssh_key="${DEV_SSH_KEY:?DEV_SSH_KEY is required}"
            server="${DEV_SERVER:?DEV_SERVER is required}"
            ;;
        prod)
            ssh_key="${PROD_SSH_KEY:?PROD_SSH_KEY is required}"
            server="${PROD_SERVER:?PROD_SERVER is required}"
            ;;
        *)
            echo "❌ Tagged deployment exists only for dev or prod." >&2
            exit 1
            ;;
    esac

    result_dir="$(realpath result)"
    if [[ "$result_dir" != /nix/store/* ]]; then
        echo "❌ EIF result must resolve to an immutable Nix store output." >&2
        exit 1
    fi
    pcr_file="$(realpath "$result_dir/pcr.json")"
    eif_file="$(realpath "$result_dir/image.eif")"
    if [[ "$pcr_file" != /nix/store/* || "$eif_file" != /nix/store/* ]]; then
        echo "❌ PCR and EIF files must remain inside the immutable Nix store." >&2
        exit 1
    fi
    test -s "$pcr_file"
    test -s "$eif_file"

    local_digest="$(openssl dgst -sha256 "$eif_file" | awk '{print $NF}')"
    if [[ ! "$local_digest" =~ ^[0-9a-f]{64}$ ]]; then
        echo "❌ Could not calculate a canonical EIF SHA-256 digest." >&2
        exit 1
    fi
    remote_eif="opensecret-${release_tag}-${environment}-${local_digest}.eif"

    printf 'Release: %s\nEnvironment: %s\nEIF SHA-256: %s\nPCR tuple:\n' \
        "$release_tag" "$environment" "$local_digest"
    jq '{PCR0, PCR1, PCR2}' "$pcr_file"
    confirmation="$environment $release_tag"
    read -r -p "Type '$confirmation' to run live publication gates and deploy: " answer
    if [[ "$answer" != "$confirmation" ]]; then
        echo "❌ Deployment cancelled." >&2
        exit 1
    fi

    # Stage first so the time-sensitive gates run immediately before the
    # remote locked replacement. The content-addressed path is harmless if a
    # later gate fails.
    scp -i "$ssh_key" "$eif_file" "${server}:~/${remote_eif}"

    just verify-legacy-pcr-propagated "$environment" "$pcr_file"
    just verify-sigstore-release-published \
        "$release_tag" \
        "$environment" \
        "$pcr_file" \
        "$eif_file"

    # One remote critical section binds the final hash check, targeted
    # termination, launch, and proxy restart. It preserves p11ne and any
    # unrelated enclave, and terminates every exact-name OpenSecret instance.
    ssh -i "$ssh_key" "$server" \
        bash -s -- "$remote_eif" "$local_digest" "$environment" "$release_tag" <<'REMOTE_DEPLOY'
    set -euo pipefail

    remote_eif="${1:?}"
    expected_digest="${2:?}"
    environment="${3:?}"
    release_tag="${4:?}"

    case "$environment" in
        dev|prod) ;;
        *) exit 1 ;;
    esac
    [[ "$release_tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || exit 1
    [[ "$expected_digest" =~ ^[0-9a-f]{64}$ ]] || exit 1
    [[ "$remote_eif" == \
        "opensecret-${release_tag}-${environment}-${expected_digest}.eif" ]] || exit 1

    for command in flock install jq nitro-cli sha256sum sudo systemctl; do
        if ! command -v "$command" >/dev/null; then
            echo "Required parent-instance command is missing: $command" >&2
            exit 1
        fi
    done

    lock_dir="$HOME/.local/state/opensecret/deploy-locks"
    install -d -m 0700 "$lock_dir"
    exec 9>"$lock_dir/host.lock"
    if ! flock -n 9; then
        echo "Another OpenSecret deployment is in progress on this parent instance." >&2
        exit 1
    fi

    remote_path="$HOME/$remote_eif"
    test -f "$remote_path"

    described="$(nitro-cli describe-enclaves)"
    jq -e 'type == "array"' <<<"$described" >/dev/null

    mapfile -t application_ids < <(
        jq -r \
            '.[] | select(.EnclaveName == "opensecret") | .EnclaveID' \
            <<<"$described"
    )
    mapfile -t ambiguous_names < <(
        jq -r \
            '.[] |
             select((.EnclaveName // "") | startswith("opensecret")) |
             select(.EnclaveName != "opensecret") |
             .EnclaveName' \
            <<<"$described"
    )
    if ((${#ambiguous_names[@]})); then
        printf 'Refusing to guess whether these enclaves are application instances: %s\n' \
            "${ambiguous_names[*]}" >&2
        exit 1
    fi

    verify_digest() {
        local actual
        actual="$(sha256sum -- "$remote_path")"
        actual="${actual%% *}"
        if [[ "$actual" != "$expected_digest" ]]; then
            echo "Staged EIF digest does not match the authenticated release." >&2
            exit 1
        fi
    }

    # Verify immediately before stopping the currently healthy application.
    verify_digest
    for enclave_id in "${application_ids[@]}"; do
        [[ -n "$enclave_id" && "$enclave_id" != "null" ]] || exit 1
        nitro-cli terminate-enclave --enclave-id "$enclave_id"
    done

    remaining="$(
        nitro-cli describe-enclaves |
            jq '[.[] | select(.EnclaveName == "opensecret")] | length'
    )"
    if [[ "$remaining" != "0" ]]; then
        echo "An OpenSecret application enclave remains after termination." >&2
        exit 1
    fi

    # Re-authenticate after termination and immediately before launch.
    verify_digest
    nitro-cli run-enclave \
        --eif-path "$remote_path" \
        --enclave-name opensecret \
        --memory 16384 \
        --cpu-count 4
    sudo -n systemctl restart socat-proxy.service
    REMOTE_DEPLOY

    echo "✅ Deployed $remote_eif with verified SHA-256 $local_digest."

# Deploy to dev using the exact checked-out, published stable tag.
deploy-dev-nix release_tag: build-eif-dev verify-pcr-dev
    just _deploy-tagged-nitro-eif {{quote(release_tag)}} dev

# Deploy to prod using the exact checked-out, published stable tag.
deploy-prod-nix release_tag: build-eif-prod verify-pcr-prod
    just _deploy-tagged-nitro-eif {{quote(release_tag)}} prod

# Deploy to preview environment without debug mode (using Nix-built EIF)
deploy-preview-nix: build-eif-preview verify-pcr-preview scp-eif-to-aws-preview
    @echo "EIF copied to preview server. Please review the PCR values and press Enter to continue with termination and deployment..."
    @read -p ""
    just terminate-enclave-preview run-eif-preview restart-socat-preview

# Clean EIF build artifacts
clean-eif:
    rm -f result
