#!/usr/bin/env bash
# Pull artifacts (Euler -> local) from an explicit whitelist of output paths.
# Never overwrites existing local files (--ignore-existing).
set -e
REMOTE="pencui@euler.ethz.ch:/cluster/project/sachan/pencui/explain-difficulty"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATHS=("outputs/results/" "log/")
for p in "${PATHS[@]}"; do
  mkdir -p "$ROOT/$p"
  rsync -avz --ignore-existing "$REMOTE/$p" "$ROOT/$p"
done
