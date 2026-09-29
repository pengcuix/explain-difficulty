#!/usr/bin/env bash
# Push code (local -> Euler). One-way; never pulls anything back.
set -e
REMOTE="pencui@euler.ethz.ch:/cluster/project/sachan/pencui/explain-difficulty/"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rsync -avz \
  --exclude '.git/' --exclude '/data/' --exclude '/log/' --exclude '/outputs/' \
  --exclude '/tmp/' --exclude '/docs/' --exclude '/.claude/' \
  --exclude '__pycache__/' --exclude '.DS_Store' \
  "$ROOT/" "$REMOTE"
