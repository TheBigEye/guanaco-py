#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-}"
TAG_PATTERN="${2:-}"

if [ -z "$OUT_DIR" ] || [ -z "$TAG_PATTERN" ]; then
  echo "USAGE: releases-to-pep-503.sh <output-dir> <tag-regex>"
  exit 0
fi

echo "==> Generating PEP 503 index:"
echo "    Output dir : $OUT_DIR"
echo "    Tag regex : $TAG_PATTERN"

mkdir -p "$OUT_DIR"

RELEASES_FILE="all_releases.txt"

if [ ! -f "$RELEASES_FILE" ]; then
  echo "WARNING: $RELEASES_FILE not found, skipping"
  exit 0
fi

MATCHING_RELEASES="$(
  jq -r '.[] | select(.tag_name | test("'"$TAG_PATTERN"'")) | .tag_name' \
    "$RELEASES_FILE" 2>/dev/null || true
)"

if [ -z "$MATCHING_RELEASES" ]; then
  echo "INFO: No matching releases for pattern $TAG_PATTERN"
  {
    echo "<!DOCTYPE html>"
    echo "<html><head><title>Simple Index</title></head><body></body></html>"
  } > "$OUT_DIR/index.html"
  exit 0
fi

echo "Matching releases:"
echo "$MATCHING_RELEASES"

{
  echo "<!DOCTYPE html>"
  echo "<html>"
  echo "  <head>"
  echo "    <meta charset=\"utf-8\">"
  echo "    <title>Simple Index</title>"
  echo "  </head>"
  echo "  <body>"
} > "$OUT_DIR/index.html"

FOUND_FILES=0

while read -r TAG; do
  echo "  Processing tag: $TAG"

  ASSETS="$(
    jq -r --arg TAG "$TAG" '
      .[] | select(.tag_name == $TAG) |
      .assets[]? |
      select(.name | endswith(".whl")) |
      "\(.browser_download_url)|\(.name)"
    ' "$RELEASES_FILE" 2>/dev/null || true
  )"

  if [ -z "$ASSETS" ]; then
    echo "    No wheel assets for $TAG"
    continue
  fi

  while IFS="|" read -r URL NAME; do
    FOUND_FILES=$((FOUND_FILES + 1))
    echo "    + $NAME"
    echo "    <a href=\"$URL\">$NAME</a><br/>" >> "$OUT_DIR/index.html"
  done <<< "$ASSETS"

done <<< "$MATCHING_RELEASES"

{
  echo "  </body>"
  echo "</html>"
} >> "$OUT_DIR/index.html"

if [ "$FOUND_FILES" -eq 0 ]; then
  echo "WARNING: Index generated but contains no wheel links"
else
  echo "SUCCESS: Generated index with $FOUND_FILES wheel(s)"
fi

exit 0
