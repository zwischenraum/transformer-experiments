#!/bin/bash
set -euo pipefail

NAMESPACE="jens"
CONTEXT="c-test"
SECRET_NAME="wandb-api-key"
JOB_MANIFEST="k8s-job.yaml"
ENV_FILE=".env"
CONFIGMAP_NAME="transformer-training-config"
CONFIG_FILE="config/training.yaml"

cleanup() {
    local reason="$1"
    echo "Cleaning up secret and configmap ($reason)..."
    kubectl --context="$CONTEXT" delete secret "$SECRET_NAME" -n "$NAMESPACE" --ignore-not-found=true
    kubectl --context="$CONTEXT" delete configmap "$CONFIGMAP_NAME" -n "$NAMESPACE" --ignore-not-found=true
}

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found"
    exit 1
fi

source "$ENV_FILE"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY not found in .env file"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found at $CONFIG_FILE"
    exit 1
fi

echo "Setting Kubernetes context to $CONTEXT..."
kubectl config use-context "$CONTEXT"

echo "Creating secret $SECRET_NAME in namespace $NAMESPACE..."
kubectl --context="$CONTEXT" create secret generic "$SECRET_NAME" \
    --from-literal=api-key="$WANDB_API_KEY" \
    -n "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl --context="$CONTEXT" apply -f -

echo "Syncing ConfigMap $CONFIGMAP_NAME from $CONFIG_FILE..."
kubectl --context="$CONTEXT" create configmap "$CONFIGMAP_NAME" \
    --from-file=training.yaml="$CONFIG_FILE" \
    -n "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl --context="$CONTEXT" apply -f -

trap 'cleanup error' ERR

echo "Applying job manifest..."
kubectl --context="$CONTEXT" apply -f "$JOB_MANIFEST"

echo "Waiting for job pods to be created and running..."
JOB_NAME="transformer-training"
for i in {1..60}; do
    POD_COUNT=$(kubectl --context="$CONTEXT" get pods -n "$NAMESPACE" -l app=transformer-training --field-selector=status.phase=Running -o name 2>/dev/null | wc -l | tr -d ' ')
    if [ "$POD_COUNT" -gt 0 ]; then
        echo "Job pods are running!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Error: Job failed to start pods within timeout"
        exit 1
    fi
    sleep 2
done

echo "Job started successfully. Pod has loaded the environment variable."
cleanup completion
trap - ERR
