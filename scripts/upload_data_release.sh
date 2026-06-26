#!/usr/bin/env bash
# Bundle and upload cached data files as a GitHub release asset.
# Run this after a full retrain to publish fresh data so others don't
# have to re-fetch everything from Baseball Reference / MLB Stats API.
#
# Usage: ./scripts/upload_data_release.sh [tag]
#   tag defaults to "data-cache"

set -euo pipefail

REPO="jackleh/MLB-Predictions"
TAG="${1:-data-cache}"
ASSET="data_cache.tar.gz"
DATA_DIR="$(dirname "$0")/../data"

echo "Bundling data cache..."
tar -czf "$ASSET" -C "$DATA_DIR" \
  $(ls "$DATA_DIR" | grep -v '\.pkl$' | grep -v 'results_cache')

echo "Asset size: $(du -sh "$ASSET" | cut -f1)"

# Delete existing release/tag if present so we can re-upload
gh release delete "$TAG" --repo "$REPO" --yes 2>/dev/null || true

echo "Creating release $TAG and uploading $ASSET..."
gh release create "$TAG" "$ASSET" \
  --repo "$REPO" \
  --title "Data Cache" \
  --notes "Cached schedule, stats, and linescore data. Download this to skip the initial data fetch on first run." \
  --prerelease

rm "$ASSET"
echo "Done. Run 'gh release download $TAG -R $REPO -D data/' to restore."
