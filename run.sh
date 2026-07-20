#!/bin/bash
set -euo pipefail

# FABv2 reads the one-task dataset prepared at /app/data/dataset.json.
TASK_ID="$2"
MODEL="$3"

fabv2 run \
  --model "$MODEL" \
  --run-id valkyrie \
  --skip-eval \
  --dataset-file /app/data/dataset.json \
  --results-dir /app/results \
  "$TASK_ID"
