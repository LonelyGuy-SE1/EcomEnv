#!/usr/bin/env bash
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.

set -euo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BOLD=''
  NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    (
      sleep "$secs"
      kill -9 "$pid" >/dev/null 2>&1 || true
    ) &
    local watcher=$!
    wait "$pid" 2>/dev/null || true
    local rc=$?
    kill "$watcher" >/dev/null 2>&1 || true
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "/tmp/${prefix}.XXXXXX"
}

CLEANUP_FILES=()
cleanup() {
  for f in "${CLEANUP_FILES[@]:-}"; do
    [ -f "$f" ] && rm -f "$f"
  done
}
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Invalid repo_dir: %s\n" "$REPO_DIR"
  exit 1
fi

PING_URL="${PING_URL%/}"

PASS=0
FAIL=0
log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { PASS=$((PASS+1)); log "${GREEN}PASSED${NC} -- $1"; }
fail() { FAIL=$((FAIL+1)); log "${RED}FAILED${NC} -- $1"; }
hint() { log "${YELLOW}HINT${NC}   -- $1"; }
stop_at() {
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Ping URL: $PING_URL"
log "Repo:     $REPO_DIR"
printf "\n"

log "${BOLD}Step 1/3: Checking Hugging Face Space health${NC} ..."
CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")

HTTP_CODE="000"
for _ in $(seq 1 10); do
  HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
    "$PING_URL/reset" --max-time 10 2>"$CURL_OUTPUT" || printf "000")
  if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "422" ]; then
    break
  fi
  sleep 2
done

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "422" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Try: curl -s -o /dev/null -w '%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "Unexpected response from Space: HTTP $HTTP_CODE"
  hint "Make sure your Space is running and the URL is correct."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."
if ! command -v docker >/dev/null 2>&1; then
  fail "docker not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

DOCKER_CONTEXT="$REPO_DIR"
DOCKERFILE_PATH=""
if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKERFILE_PATH="$REPO_DIR/Dockerfile"
elif [ -f "$REPO_DIR/ecom/server/Dockerfile" ]; then
  DOCKERFILE_PATH="$REPO_DIR/ecom/server/Dockerfile"
  DOCKER_CONTEXT="$REPO_DIR/ecom"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKERFILE_PATH="$REPO_DIR/server/Dockerfile"
  DOCKER_CONTEXT="$REPO_DIR"
else
  fail "Could not find Dockerfile (checked root, ecom/server, and server/)"
  stop_at "Step 2"
fi

log "  Found Dockerfile: $DOCKERFILE_PATH"
BUILD_OK=false
if run_with_timeout "$DOCKER_BUILD_TIMEOUT" \
  docker build -f "$DOCKERFILE_PATH" "$DOCKER_CONTEXT" >/dev/null 2>&1; then
  BUILD_OK=true
fi

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."
if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv CLI not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=""
VALIDATE_TARGET="$REPO_DIR/ecom"
if [ -f "$REPO_DIR/openenv.yaml" ]; then
  VALIDATE_TARGET="$REPO_DIR"
fi

if VALIDATE_OUTPUT="$(openenv validate "$VALIDATE_TARGET" 2>&1)"; then
  VALIDATE_OK=true
fi

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
