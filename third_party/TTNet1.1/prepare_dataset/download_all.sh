#!/bin/bash
# Download / resume TTNet dataset with retry and resume support.
BASE="https://lab.osai.ai/datasets/openttgames/data"
ROOT="G:/workbuddy_project/TTnet/TTNet/dataset"

TRAIN_VID="$ROOT/training/videos"
TRAIN_ANN="$ROOT/training/annotations"
TEST_VID="$ROOT/test/videos"
TEST_ANN="$ROOT/test/annotations"

mkdir -p "$TRAIN_VID" "$TRAIN_ANN" "$TEST_VID" "$TEST_ANN"

dl() {
  local url="$1" out="$2" name local_sz remote_sz
  name="$(basename "$out")"
  local_sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
  remote_sz=$(curl -sSI -m 30 "$url" 2>/dev/null | grep -i "^content-length" | tail -1 | awk '{print $2}' | tr -d '\r')
  remote_sz=${remote_sz:-0}
  if [ "$local_sz" -eq "$remote_sz" ] && [ "$remote_sz" -gt 0 ]; then
    echo "OK(complete): $name"
    return
  fi
  echo "Downloading $name (local $((local_sz/1024/1024))MB / remote $((remote_sz/1024/1024))MB)..."
  curl -sSL -C - --connect-timeout 30 --retry 200 --retry-delay 3 --retry-all-errors -o "$out" "$url"
  if [ $? -eq 0 ]; then
    echo "DONE: $name"
  else
    echo "FAILED: $name"
  fi
}

# training videos
for i in 1 2 3 4 5; do dl "$BASE/game_${i}.mp4" "$TRAIN_VID/game_${i}.mp4"; done
# training annotations
for i in 1 2 3 4 5; do dl "$BASE/game_${i}.zip" "$TRAIN_ANN/game_${i}.zip"; done
# test videos
for i in 1 2 3 4 5 6 7; do dl "$BASE/test_${i}.mp4" "$TEST_VID/test_${i}.mp4"; done
# test annotations
for i in 1 2 3 4 5 6 7; do dl "$BASE/test_${i}.zip" "$TEST_ANN/test_${i}.zip"; done

echo "=== ALL DONE ==="
