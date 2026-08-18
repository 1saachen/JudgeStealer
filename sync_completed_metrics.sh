#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs}"
DEST_ROOT="${DEST_ROOT:-$ROOT/outputs}"
MODE=compact

usage() {
  cat >&2 <<'EOF'
usage: sync_completed_metrics.sh [--all-metrics]

Default: copy only metrics_compact.json from completed runs.
--all-metrics: copy every metrics*.json file from completed runs.
Environment overrides: SOURCE_ROOT=/path DEST_ROOT=/path
EOF
}

case "${1:-}" in
  "") ;;
  --all-metrics) MODE=all ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "source output directory does not exist: $SOURCE_ROOT" >&2
  exit 1
fi
mkdir -p "$DEST_ROOT"

completed=0
copied=0
skipped=0

copy_metric() {
  local source_file="$1" run_dir="$2" name dest_dir dest_file
  name="$(basename "$run_dir")"
  dest_dir="$DEST_ROOT/$name"
  dest_file="$dest_dir/$(basename "$source_file")"
  mkdir -p "$dest_dir"

  if [[ -e "$dest_file" && "$source_file" -ef "$dest_file" ]]; then
    skipped=$((skipped + 1))
    return 0
  fi
  cp -p -- "$source_file" "$dest_file"
  copied=$((copied + 1))
}

while IFS= read -r -d '' compact_metric; do
  run_dir="$(dirname "$compact_metric")"
  completed=$((completed + 1))

  if [[ "$MODE" == compact ]]; then
    copy_metric "$compact_metric" "$run_dir"
    continue
  fi

  while IFS= read -r -d '' metric_file; do
    copy_metric "$metric_file" "$run_dir"
  done < <(find "$run_dir" -maxdepth 1 -type f -name 'metrics*.json' -print0)
done < <(
  find "$SOURCE_ROOT" -mindepth 2 -maxdepth 2 -type f \
    -name 'metrics_compact.json' -print0 | sort -z
)

echo "completed_runs=$completed copied_metrics=$copied skipped_metrics=$skipped"
echo "destination=$DEST_ROOT mode=$MODE"
