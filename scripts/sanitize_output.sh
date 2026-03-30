#!/usr/bin/env bash
# sanitize_output.sh — Strip private information from output directory before packaging
# Used by botrans GitHub Actions workflow before uploading output.tar.gz
#
# Targets all text files (json, txt, log, yaml, yml, md, csv, tsv)
# Skips binary files (epub, zip, png, jpg, etc.)

set -euo pipefail

OUTPUT_DIR="${1:-output}"

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Usage: $0 [output_dir]"
  echo "Error: directory '$OUTPUT_DIR' does not exist"
  exit 1
fi

echo "[sanitize] Scanning $OUTPUT_DIR ..."

# 1. Delete files that should never be in the package
find "$OUTPUT_DIR" -name ".DS_Store" -delete 2>/dev/null || true
find "$OUTPUT_DIR" -name "config.yaml" -delete 2>/dev/null || true
find "$OUTPUT_DIR" -name "config_*.yaml" -delete 2>/dev/null || true
rm -f config.yaml config_*.yaml 2>/dev/null || true

# 2. Remove input.epub (user already has it, saves space for resume)
find "$OUTPUT_DIR" -name "input.epub" -delete 2>/dev/null || true
find "$OUTPUT_DIR" -name "input_original.pdf" -delete 2>/dev/null || true

COUNT=0

# 3. Sanitize all text files (using process substitution to keep COUNT in parent shell)
while read -r f; do
  # API keys — various formats
  sed -i.bak 's/api_key["'\'']*[:=][[:space:]]*["'\'']*[A-Za-z0-9_-]\{20,\}/api_key: [REDACTED]/g' "$f"
  # OpenAI/DeepSeek-style keys (sk-..., with or without quotes)
  sed -i.bak 's/sk-[A-Za-z0-9_-]\{20,\}/[REDACTED]/g' "$f"
  # Anthropic-style keys
  sed -i.bak 's/sk-ant-[A-Za-z0-9_-]\{20,\}/[REDACTED]/g' "$f"
  # UUID-style keys (like the proxy mapped keys)
  sed -i.bak 's/api_key["'\'']*[:=][[:space:]]*["'\'']*[0-9a-f]\{8\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{12\}/api_key: [REDACTED]/g' "$f"

  # Bearer tokens
  sed -i.bak 's/Bearer [A-Za-z0-9_.-]\{20,\}/Bearer [REDACTED]/g' "$f"

  # Authorization headers
  sed -i.bak 's/Authorization["'\'']*[:=][[:space:]]*["'\'']*[A-Za-z0-9_.-]\{20,\}/Authorization: [REDACTED]/g' "$f"

  # Proxy domains — strip full URLs
  sed -i.bak 's|https\{0,1\}://[a-zA-Z0-9_-]*\.shenshei\.fans[^ "'\'']*|[REDACTED_URL]|g' "$f"
  sed -i.bak 's|https\{0,1\}://[a-zA-Z0-9_-]*\.zzhou\.info[^ "'\'']*|[REDACTED_URL]|g' "$f"

  # Local paths — macOS, Linux, Windows
  sed -i.bak 's|/Users/[^ "'\'']*|[REDACTED_PATH]|g' "$f"
  sed -i.bak 's|/home/[^ "'\'']*|[REDACTED_PATH]|g' "$f"
  sed -i.bak 's|/root/[^ "'\'']*|[REDACTED_PATH]|g' "$f"
  sed -i.bak 's|C:\\Users\\[^ "'\'']*|[REDACTED_PATH]|g' "$f"

  # Clean up sed backup files
  rm -f "${f}.bak"

  COUNT=$((COUNT + 1))
done < <(find "$OUTPUT_DIR" -type f \( \
  -name "*.json" -o -name "*.txt" -o -name "*.log" \
  -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" \
  -o -name "*.csv" -o -name "*.tsv" \
\))

echo "[sanitize] Processed $COUNT text files"
echo "[sanitize] Done"
