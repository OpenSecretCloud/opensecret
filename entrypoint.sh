#!/bin/bash

set -e

# Function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting entrypoint script"

# Start the logging script
log "Starting log exports"

# Redirect all output to the logging script via VSOCK
exec > >(socat - VSOCK-CONNECT:3:8011) 2>&1

# Read and set APP_MODE from file
log "Reading /app/APP_MODE"
if [ -f /app/APP_MODE ]; then
    APP_MODE=""
    APP_MODE="$(cat /app/APP_MODE)" || { log "Failed to read /app/APP_MODE"; exit 1; }
    export APP_MODE
    log "Set APP_MODE=$APP_MODE from /app/APP_MODE"
else
    log "ERROR: /app/APP_MODE is missing. Please ensure the file exists and contains a valid mode (dev/preview/prod/custom)"
    exit 1
fi

log "Starting entrypoint script"
log "APP_MODE=$APP_MODE"

# Configure loopback interface
log "Configuring loopback interface"
ip addr add 127.0.0.1/8 dev lo
ip link set dev lo up

# Function to send request and receive response via VSOCK
vsock_request() {
    local cid=$1
    local port=$2
    local request=$3

    response=$(python3 /app/vsock_helper.py "$cid" "$port" "$request")
    
    # Check if the response contains an error
    if echo "$response" | jq -e 'has("error")' > /dev/null; then
        error_message=$(echo "$response" | jq -r '.error')
        log "VSOCK request failed: $error_message"
        return 1
    fi

    echo "$response"
}

# Function to get AWS credentials
get_aws_credentials() {
    local cid=3
    local port=8003
    local request='{"request_type":"credentials","key_name":null}'

    vsock_request $cid $port "$request"
}

# Function to get secret from Secrets Manager
get_database_url_secret() {
    local cid=3
    local port=8003

    # Determine the correct secret name based on APP_MODE
    local secret_name
    if [ "$APP_MODE" = "prod" ]; then
        secret_name="opensecret_prod_database_url"
    elif [ "$APP_MODE" = "preview" ]; then
        secret_name="opensecret_preview1_database_url"
    elif [ "$APP_MODE" = "custom" ]; then
        if [ -z "$ENV_NAME" ]; then
            log "Error: ENV_NAME must be set when using custom mode"
            exit 1
        fi
        secret_name="opensecret_${ENV_NAME}_database_url"
    else
        secret_name="opensecret_dev_database_url"
    fi

    local request="{\"request_type\":\"SecretsManager\",\"key_name\":\"$secret_name\"}"

    vsock_request $cid $port "$request"
}

# Function to get secret from Secrets Manager
get_continuum_proxy_api_key_secret() {
    local cid=3
    local port=8003

    # Determine the correct secret name based on APP_MODE
    local secret_name
    if [ "$APP_MODE" = "prod" ]; then
        secret_name="continuum_proxy_prod_api_key"
    elif [ "$APP_MODE" = "preview" ]; then
        secret_name="continuum_proxy_preview1_api_key"
    elif [ "$APP_MODE" = "custom" ]; then
        if [ -z "$ENV_NAME" ]; then
            log "Error: ENV_NAME must be set when using custom mode"
            exit 1
        fi
        secret_name="continuum_proxy_${ENV_NAME}_api_key"
    else
        secret_name="continuum_proxy_dev_api_key"
    fi

    local request="{\"request_type\":\"SecretsManager\",\"key_name\":\"$secret_name\"}"

    vsock_request $cid $port "$request"
}

# Function to get tinfoil proxy API key from Secrets Manager
get_tinfoil_proxy_api_key_secret() {
    local cid=3
    local port=8003

    # Determine the correct secret name based on APP_MODE
    local secret_name
    if [ "$APP_MODE" = "prod" ]; then
        secret_name="tinfoil_proxy_prod_api_key"
    elif [ "$APP_MODE" = "preview" ]; then
        secret_name="tinfoil_proxy_preview1_api_key"
    elif [ "$APP_MODE" = "custom" ]; then
        if [ -z "$ENV_NAME" ]; then
            log "Error: ENV_NAME must be set when using custom mode"
            exit 1
        fi
        secret_name="tinfoil_proxy_${ENV_NAME}_api_key"
    else
        secret_name="tinfoil_proxy_dev_api_key"
    fi

    local request="{\"request_type\":\"SecretsManager\",\"key_name\":\"$secret_name\"}"

    vsock_request $cid $port "$request"
}

# Get AWS credentials
log "Fetching AWS credentials"
aws_creds=$(get_aws_credentials)
if [ -z "$aws_creds" ]; then
    log "Error: Failed to get AWS credentials"
    exit 1
fi

# Add error checking for jq parsing
if ! access_key_id=$(echo "$aws_creds" | jq -r '.response_value.AccessKeyId'); then
    log "Error: Failed to parse AccessKeyId from AWS credentials"
    log "AWS credentials response: $aws_creds"
    exit 1
fi
if ! secret_access_key=$(echo "$aws_creds" | jq -r '.response_value.SecretAccessKey'); then
    log "Error: Failed to parse SecretAccessKey from AWS credentials"
    exit 1
fi
if ! session_token=$(echo "$aws_creds" | jq -r '.response_value.Token'); then
    log "Error: Failed to parse Token from AWS credentials"
    exit 1
fi
if ! region=$(echo "$aws_creds" | jq -r '.response_value.Region'); then
    log "Error: Failed to parse Region from AWS credentials"
    exit 1
fi

log "AWS credentials retrieved and parsed successfully"

# Get encrypted database URL from Secrets Manager
log "Fetching encrypted database URL"
secret_response=$(get_database_url_secret)
log "Retrieved raw secret response"

# Extract the database_url value from the JSON structure
encrypted_db_url=$(echo "$secret_response" | jq -r '.response_value | fromjson | .database_url')
if [ -z "$encrypted_db_url" ]; then
    log "Error: Failed to get encrypted database URL"
    log "Secret response: $secret_response"
    exit 1
fi

log "Encrypted database URL retrieved successfully"

# Decrypt the database URL using kmstool_enclave_cli
log "Decrypting database URL"
decryption_output=$(kmstool_enclave_cli decrypt \
    --region "$region" \
    --proxy-port 8000 \
    --aws-access-key-id "$access_key_id" \
    --aws-secret-access-key "$secret_access_key" \
    --aws-session-token "$session_token" \
    --ciphertext "$encrypted_db_url" 2>&1)

log "Got decryption output, parsing URL"

decrypted_db_url=$(echo "$decryption_output" | sed -n 's/PLAINTEXT: //p')

if [ -z "$decrypted_db_url" ]; then
    log "Error: Failed to decrypt database URL"
    log "Decryption output: $decryption_output"
    exit 1
fi

log "Database URL decrypted successfully"

# Decode the base64 decrypted URL
decoded_db_url=$(echo "$decrypted_db_url" | base64 -d)

if [ -z "$decoded_db_url" ]; then
    log "Error: Failed to decode base64 database URL"
    exit 1
fi

# Extract the hostname from the decoded DATABASE_URL and add it to /etc/hosts
DB_HOSTNAME=$(echo "$decoded_db_url" | sed -n 's/.*@\(.*\)\/.*/\1/p')
if [ -z "$DB_HOSTNAME" ]; then
    log "Error: Failed to extract DB_HOSTNAME from decoded URL"
    exit 1
fi

echo "127.0.0.1 $DB_HOSTNAME" >> /etc/hosts
log "Added $DB_HOSTNAME to /etc/hosts"

# Add OpenAI API hostname to /etc/hosts
echo "127.0.0.1 api.openai.com" >> /etc/hosts
log "Added api.openai.com to /etc/hosts"

# Add Resend API hostname to /etc/hosts
echo "127.0.0.8 api.resend.com" >> /etc/hosts
log "Added api.resend.com to /etc/hosts"

# Add continuum hostnames to /etc/hosts
echo "127.0.0.2 api.privatemode.ai" >> /etc/hosts
echo "127.0.0.3 cdn.confidential.cloud" >> /etc/hosts
echo "127.0.0.4 secret.privatemode.ai" >> /etc/hosts
echo "127.0.0.5 coordinator.privatemode.ai" >> /etc/hosts
echo "127.0.0.6 kdsintf.amd.com" >> /etc/hosts

log "Added privatemode.ai, confidential.cloud, and AMD domains to /etc/hosts"

# Add GitHub OAuth hostnames to /etc/hosts
echo "127.0.0.9 github.com" >> /etc/hosts
echo "127.0.0.10 api.github.com" >> /etc/hosts
log "Added GitHub OAuth domains to /etc/hosts"

# Add Google OAuth hostnames to /etc/hosts
echo "127.0.0.11 oauth2.googleapis.com" >> /etc/hosts
echo "127.0.0.12 www.googleapis.com" >> /etc/hosts
log "Added Google OAuth domains to /etc/hosts"

# Add Apple OAuth hostname to /etc/hosts
echo "127.0.0.15 appleid.apple.com" >> /etc/hosts
log "Added Apple OAuth domain to /etc/hosts"

# Add AWS SQS hostname to /etc/hosts
echo "127.0.0.13 sqs.us-east-2.amazonaws.com" >> /etc/hosts
log "Added AWS SQS domain to /etc/hosts"

# Add billing hostname to /etc/hosts based on APP_MODE
if [ "$APP_MODE" = "prod" ]; then
    echo "127.0.0.14 billing.opensecret.cloud" >> /etc/hosts
    log "Added production billing domain to /etc/hosts"
else
    echo "127.0.0.14 billing-dev.opensecret.cloud" >> /etc/hosts
    log "Added development billing domain to /etc/hosts"
fi

# Add Tinfoil proxy hostnames to /etc/hosts
echo "127.0.0.16 api-github-proxy.tinfoil.sh" >> /etc/hosts
echo "127.0.0.17 tuf-repo-cdn.sigstore.dev" >> /etc/hosts
echo "127.0.0.18 deepseek-r1-70b-p.model.tinfoil.sh" >> /etc/hosts
echo "127.0.0.19 kds-proxy.tinfoil.sh" >> /etc/hosts
echo "127.0.0.20 gh-attestation-proxy.tinfoil.sh" >> /etc/hosts
echo "127.0.0.21 doc-upload.model.tinfoil.sh" >> /etc/hosts
log "Added Tinfoil proxy domains to /etc/hosts"

touch /app/libnsm.so
log "Created /app/libnsm.so"

# Print network information for debugging
log "Network configuration:"
ip addr show
ip route
cat /etc/hosts

# Start the traffic forwarder for the database in the background
log "Starting database traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.1 5432 3 8001 &

# Start the traffic forwarder for OpenAI API in the background
log "Starting OpenAI API traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.1 443 3 8002 &

# Start the traffic forwarder for Resend API in the background
log "Starting Resend API traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.8 443 3 8010 &

# Start the traffic forwarder for Continuum API in the background
log "Starting Continuum API traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.2 443 3 8004 &

# Start the traffic forwarder for Continuum CDN in the background
log "Starting Continuum CDN traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.3 443 3 8005 &

# Start the traffic forwarder for Continuum Secret Service in the background
log "Starting Continuum Secret Service traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.4 443 3 8006 &

# Start the traffic forwarder for Continuum Coordinator in the background
log "Starting Continuum Coordinator traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.5 443 3 8007 &

# Start the traffic forwarder for AMD KDS Interface in the background
log "Starting AMD KDS Interface traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.6 443 3 8008 &

# Start the traffic forwarder for GitHub in the background
log "Starting GitHub traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.9 443 3 8012 &

# Start the traffic forwarder for GitHub API in the background
log "Starting GitHub API traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.10 443 3 8013 &

# Start the traffic forwarder for Google OAuth in the background
log "Starting Google OAuth traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.11 443 3 8014 &

# Start the traffic forwarder for Google APIs in the background
log "Starting Google APIs traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.12 443 3 8015 &

# Start the traffic forwarder for AWS SQS in the background
log "Starting AWS SQS traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.13 443 3 8016 &

# Start the traffic forwarder for billing service in the background
log "Starting billing service traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.14 443 3 8017 &

# Start the traffic forwarder for Apple OAuth in the background
log "Starting Apple OAuth traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.15 443 3 8018 &

# Start the traffic forwarders for Tinfoil proxy in the background
log "Starting Tinfoil API GitHub proxy traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.16 443 3 8019 &

log "Starting TUF Repository CDN traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.17 443 3 8020 &

log "Starting Tinfoil DeepSeek model traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.18 443 3 8021 &

log "Starting Tinfoil KDS proxy traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.19 443 3 8022 &

log "Starting Tinfoil GitHub proxy traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.20 443 3 8023 &

log "Starting Tinfoil Document Upload traffic forwarder"
python3 /app/traffic_forwarder.py 127.0.0.21 443 3 8024 &

# Wait for the forwarders to start
log "Waiting for forwarders to start"
sleep 5

# Test the connection to PostgreSQL
log "Testing connection to PostgreSQL:"
if timeout 5 bash -c '</dev/tcp/127.0.0.1/5432'; then
    log "PostgreSQL connection successful"
else
    log "PostgreSQL connection failed"
fi

# Test the connection to OpenAI API (Note: This will only test if the port is open)
log "Testing connection to OpenAI API:"
if timeout 5 bash -c '</dev/tcp/127.0.0.1/443'; then
    log "OpenAI API connection successful"
else
    log "OpenAI API connection failed"
fi

# Test the connection to Continuum API
log "Testing connection to Continuum API:"
if timeout 5 bash -c '</dev/tcp/127.0.0.2/443'; then
    log "Continuum API connection successful"
else
    log "Continuum API connection failed"
fi

log "Testing connection to Continuum CDN:"
if timeout 5 bash -c '</dev/tcp/127.0.0.3/443'; then
    log "Continuum CDN connection successful"
else
    log "Continuum CDN connection failed"
fi

log "Testing connection to Continuum Secret Service:"
if timeout 5 bash -c '</dev/tcp/127.0.0.4/443'; then
    log "Continuum Secret Service connection successful"
else
    log "Continuum Secret Service connection failed"
fi

log "Testing connection to Continuum Coordinator:"
if timeout 5 bash -c '</dev/tcp/127.0.0.5/443'; then
    log "Continuum Coordinator connection successful"
else
    log "Continuum Coordinator connection failed"
fi

log "Testing connection to AMD KDS Interface:"
if timeout 5 bash -c '</dev/tcp/127.0.0.6/443'; then
    log "AMD KDS Interface connection successful"
else
    log "AMD KDS Interface connection failed"
fi

# Test the connection to GitHub
log "Testing connection to GitHub:"
if timeout 5 bash -c '</dev/tcp/127.0.0.9/443'; then
    log "GitHub connection successful"
else
    log "GitHub connection failed"
fi

# Test the connection to GitHub API
log "Testing connection to GitHub API:"
if timeout 5 bash -c '</dev/tcp/127.0.0.10/443'; then
    log "GitHub API connection successful"
else
    log "GitHub API connection failed"
fi

# Test the connection to Google OAuth
log "Testing connection to Google OAuth:"
if timeout 5 bash -c '</dev/tcp/127.0.0.11/443'; then
    log "Google OAuth connection successful"
else
    log "Google OAuth connection failed"
fi

# Test the connection to Google APIs
log "Testing connection to Google APIs:"
if timeout 5 bash -c '</dev/tcp/127.0.0.12/443'; then
    log "Google APIs connection successful"
else
    log "Google APIs connection failed"
fi

# Test the connection to AWS SQS
log "Testing connection to AWS SQS:"
if timeout 5 bash -c '</dev/tcp/127.0.0.13/443'; then
    log "AWS SQS connection successful"
else
    log "AWS SQS connection failed"
fi

# Test the connection to billing service
log "Testing connection to billing service:"
if timeout 5 bash -c '</dev/tcp/127.0.0.14/443'; then
    log "Billing service connection successful"
else
    log "Billing service connection failed"
fi

# Test the connection to Apple OAuth
log "Testing connection to Apple OAuth:"
if timeout 5 bash -c '</dev/tcp/127.0.0.15/443'; then
    log "Apple OAuth connection successful"
else
    log "Apple OAuth connection failed"
fi

# Test the connections to Tinfoil proxy services
log "Testing connection to Tinfoil API GitHub proxy:"
if timeout 5 bash -c '</dev/tcp/127.0.0.16/443'; then
    log "Tinfoil API GitHub proxy connection successful"
else
    log "Tinfoil API GitHub proxy connection failed"
fi

log "Testing connection to TUF Repository CDN:"
if timeout 5 bash -c '</dev/tcp/127.0.0.17/443'; then
    log "TUF Repository CDN connection successful"
else
    log "TUF Repository CDN connection failed"
fi

log "Testing connection to Tinfoil DeepSeek model:"
if timeout 5 bash -c '</dev/tcp/127.0.0.18/443'; then
    log "Tinfoil DeepSeek model connection successful"
else
    log "Tinfoil DeepSeek model connection failed"
fi

log "Testing connection to Tinfoil KDS proxy:"
if timeout 5 bash -c '</dev/tcp/127.0.0.19/443'; then
    log "Tinfoil KDS proxy connection successful"
else
    log "Tinfoil KDS proxy connection failed"
fi

log "Testing connection to Tinfoil GitHub proxy:"
if timeout 5 bash -c '</dev/tcp/127.0.0.20/443'; then
    log "Tinfoil GitHub proxy connection successful"
else
    log "Tinfoil GitHub proxy connection failed"
fi

log "Testing connection to Tinfoil Document Upload:"
if timeout 5 bash -c '</dev/tcp/127.0.0.21/443'; then
    log "Tinfoil Document Upload connection successful"
else
    log "Tinfoil Document Upload connection failed"
fi

# Start the continuum-proxy if we're in AWS Nitro mode
if [ "$APP_MODE" != "local" ]; then
    # Get Continuum Proxy API key from Secrets Manager
    log "Fetching Continuum Proxy API key"
    continuum_proxy_api_key_response=$(get_continuum_proxy_api_key_secret)
    log "Retrieved raw Continuum Proxy API key response"

    # Check if the response is an error
    if echo "$continuum_proxy_api_key_response" | jq -e '.response_type == "error"' > /dev/null; then
        error_message=$(echo "$continuum_proxy_api_key_response" | jq -r '.response_value')
        log "Error: Failed to get Continuum Proxy API key. Error message: $error_message"
        exit 1
    fi

    # Extract the encrypted API key value from the JSON structure
    continuum_proxy_api_key_encrypted=$(echo "$continuum_proxy_api_key_response" | jq -r '.response_value | fromjson | .api_key')
    if [ -z "$continuum_proxy_api_key_encrypted" ]; then
        log "Error: Failed to extract Continuum Proxy API key from the response"
        log "Secret response: $continuum_proxy_api_key_response"
        exit 1
    fi

    # Decrypt the API key using kmstool_enclave_cli
    log "Decrypting Continuum Proxy API key"
    decryption_output=$(kmstool_enclave_cli decrypt \
        --region "$region" \
        --proxy-port 8000 \
        --aws-access-key-id "$access_key_id" \
        --aws-secret-access-key "$secret_access_key" \
        --aws-session-token "$session_token" \
        --ciphertext "$continuum_proxy_api_key_encrypted" 2>&1)

    decrypted_api_key=$(echo "$decryption_output" | sed -n 's/PLAINTEXT: //p')

    if [ -z "$decrypted_api_key" ]; then
        log "Error: Failed to decrypt Continuum Proxy API key"
        log "Decryption output: $decryption_output"
        exit 1
    fi

    # Base64 decode the decrypted API key
    continuum_proxy_api_key=$(echo "$decrypted_api_key" | base64 -d)

    if [ -z "$continuum_proxy_api_key" ]; then
        log "Error: Failed to base64 decode Continuum Proxy API key"
        exit 1
    fi

    log "Continuum Proxy API key retrieved, decrypted, and decoded successfully"

    log "Starting continuum-proxy on port 8092"
    /app/continuum-proxy --port 8092 --apiKey "$continuum_proxy_api_key" &

    # Wait for the proxy to start
    sleep 5

    # Set OPENAI_API_BASE to point to the local proxy
    export OPENAI_API_BASE="http://127.0.0.1:8092"
    
    # Also start tinfoil-proxy
    # Get Tinfoil Proxy API key from Secrets Manager
    log "Fetching Tinfoil Proxy API key"
    tinfoil_proxy_api_key_response=$(get_tinfoil_proxy_api_key_secret)
    log "Retrieved raw Tinfoil Proxy API key response"

    # Check if the response is an error
    if echo "$tinfoil_proxy_api_key_response" | jq -e '.response_type == "error"' > /dev/null; then
        error_message=$(echo "$tinfoil_proxy_api_key_response" | jq -r '.response_value')
        log "Error: Failed to get Tinfoil Proxy API key. Error message: $error_message"
        exit 1
    fi

    # Extract the encrypted API key value from the JSON structure
    tinfoil_proxy_api_key_encrypted=$(echo "$tinfoil_proxy_api_key_response" | jq -r '.response_value | fromjson | .api_key')
    if [ -z "$tinfoil_proxy_api_key_encrypted" ]; then
        log "Error: Failed to extract Tinfoil Proxy API key from the response"
        log "Secret response: $tinfoil_proxy_api_key_response"
        exit 1
    fi

    # Decrypt the API key using kmstool_enclave_cli
    log "Decrypting Tinfoil Proxy API key"
    decryption_output=$(kmstool_enclave_cli decrypt \
        --region "$region" \
        --proxy-port 8000 \
        --aws-access-key-id "$access_key_id" \
        --aws-secret-access-key "$secret_access_key" \
        --aws-session-token "$session_token" \
        --ciphertext "$tinfoil_proxy_api_key_encrypted" 2>&1)

    decrypted_api_key=$(echo "$decryption_output" | sed -n 's/PLAINTEXT: //p')

    if [ -z "$decrypted_api_key" ]; then
        log "Error: Failed to decrypt Tinfoil Proxy API key"
        log "Decryption output: $decryption_output"
        exit 1
    fi

    # Base64 decode the decrypted API key
    tinfoil_proxy_api_key=$(echo "$decrypted_api_key" | base64 -d)

    if [ -z "$tinfoil_proxy_api_key" ]; then
        log "Error: Failed to base64 decode Tinfoil Proxy API key"
        exit 1
    fi

    log "Tinfoil Proxy API key retrieved, decrypted, and decoded successfully"

    # Set environment variable for tinfoil-proxy and start it
    log "Starting tinfoil-proxy on port 8093"
    TINFOIL_API_KEY="$tinfoil_proxy_api_key" TINFOIL_PROXY_PORT=8093 /app/tinfoil-proxy &

    # Wait for the proxy to start
    sleep 5
    
    # Set TINFOIL_API_BASE for nitro mode
    export TINFOIL_API_BASE="http://127.0.0.1:8093"
else
    # For local mode, use the default OpenAI API base or the one set in the environment
    export OPENAI_API_BASE=${OPENAI_API_BASE:-"https://api.openai.com"}
    # No tinfoil proxy in local mode
    export TINFOIL_API_BASE=""
fi

# Start the opensecret
log "Starting opensecret..."
RUST_LOG_STYLE=never RUST_LOG=debug APP_MODE="$APP_MODE" OPENAI_API_BASE="$OPENAI_API_BASE" TINFOIL_API_BASE="$TINFOIL_API_BASE" /app/opensecret &

# Wait for the opensecret to start
log "Waiting for opensecret to start"
sleep 5

# Start socat to forward from vsock to the opensecret
log "Starting socat..."
socat VSOCK-LISTEN:5000,reuseaddr,fork TCP:0.0.0.0:3000
