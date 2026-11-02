#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/scripts/run_high_vram_server.sh" "$@"
