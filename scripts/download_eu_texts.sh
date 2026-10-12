#!/usr/bin/env bash
# Download the official EU AI Act and GDPR texts (XHTML) from the EU Publications Office
# machine-readable endpoint (Cellar). EU legislation may be reused (Commission Decision 2011/833/EU).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
for celex in 32024R1689 32016R0679; do
  curl -sSL --fail -H "Accept: application/xhtml+xml" -H "Accept-Language: eng" \
       -o "data/raw/${celex}.xhtml" "http://publications.europa.eu/resource/celex/${celex}"
  echo "downloaded data/raw/${celex}.xhtml ($(wc -c < "data/raw/${celex}.xhtml") bytes)"
done
