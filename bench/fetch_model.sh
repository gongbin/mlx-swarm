#!/bin/sh
# Download an mlx-community model from ModelScope into models/<name>.
# Hugging Face is too slow on our network (10-30 KB/s); ModelScope mirrors mlx-community.
#   sh bench/fetch_model.sh Qwen3-8B-4bit [org/Name ...]   (org defaults to mlx-community)
set -e
cd "$(dirname "$0")/.."
for repo in "$@"; do
  case "$repo" in */*) ;; *) repo="mlx-community/$repo" ;; esac
  name="${repo#*/}"
  dir="models/$name"
  mkdir -p "$dir"
  files=$(curl -sf "https://modelscope.cn/api/v1/models/$repo/repo/files?Recursive=true" |
    python3 -c "import json,sys; print(' '.join(f['Path'] for f in json.load(sys.stdin)['Data']['Files']
      if f['Type'] == 'blob' and f['Path'].endswith(('.json', '.safetensors', '.txt', '.jinja', '.py'))))")
  for f in $files; do
    echo "$name/$f"
    mkdir -p "$dir/$(dirname "$f")"
    curl -sSfL --retry 5 -C - -o "$dir/$f" \
      "https://modelscope.cn/models/$repo/resolve/master/$f"
  done
done
