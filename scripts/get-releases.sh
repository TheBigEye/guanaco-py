#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${REPO:-TheBigEye/llama-cpp-python-cpu}"
API_URL="https://api.github.com/repos/${REPO}/releases?per_page=100"
mkdir -p index/whl
curl -fsSL "$API_URL" -o all_releases.txt || true
jq -r '.[].tag_name' all_releases.txt 2>/dev/null | sed '/^$/d' > all_releases_tags.txt || true
echo "FOUND_TAGS=$(wc -l < all_releases_tags.txt || echo 0)"
exit 0
