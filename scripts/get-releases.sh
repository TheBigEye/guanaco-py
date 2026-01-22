#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Fetching releases from GitHub API"

REPO="${GITHUB_REPOSITORY:-TheBigEye/llama-cpp-python-cpu}"
API_URL="https://api.github.com/repos/${REPO}/releases"

mkdir -p index/whl

RELEASES_JSON="$(curl -fsSL "$API_URL" || true)"

if [ -z "$RELEASES_JSON" ]; then
  echo "WARNING: GitHub API returned empty response"
fi

echo "$RELEASES_JSON" > all_releases.txt || true

TAGS="$(echo "$RELEASES_JSON" | jq -r '.[].tag_name' 2>/dev/null || true)"

if [ -z "$TAGS" ]; then
  echo "WARNING: No releases found"
else
  echo "Found releases:"
  echo "$TAGS"
fi

echo "==> Generating PEP 503 indices"

./scripts/releases-to-pep-503.sh index/whl/cpu    '^[v]?[0-9]+\.[0-9]+\.[0-9]+$'        || true
./scripts/releases-to-pep-503.sh index/whl/cu121  '^[v]?[0-9]+\.[0-9]+\.[0-9]+-cu121$'  || true
./scripts/releases-to-pep-503.sh index/whl/cu122  '^[v]?[0-9]+\.[0-9]+\.[0-9]+-cu122$'  || true
./scripts/releases-to-pep-503.sh index/whl/cu123  '^[v]?[0-9]+\.[0-9]+\.[0-9]+-cu123$'  || true
./scripts/releases-to-pep-503.sh index/whl/cu124  '^[v]?[0-9]+\.[0-9]+\.[0-9]+-cu124$'  || true
./scripts/releases-to-pep-503.sh index/whl/metal  '^[v]?[0-9]+\.[0-9]+\.[0-9]+-metal$'  || true

CPU_INDEX="index/whl/cpu"

if [ ! -d "$CPU_INDEX" ]; then
  echo "ERROR: CPU index directory was not created"
  exit 0
fi

CPU_FILES="$(find "$CPU_INDEX" -type f | wc -l | tr -d ' ')"

if [ "$CPU_FILES" -eq 0 ]; then
  echo "WARNING: CPU index exists but contains no files"
else
  echo "SUCCESS: CPU index generated with $CPU_FILES files"
fi

echo "==> Index generation complete"
exit 0
