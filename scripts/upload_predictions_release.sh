#!/usr/bin/env bash
# Bundle and upload the versioned prediction archive as a GitHub release asset.
# Run this after a retrain (which forks a new model version) so the frozen
# predictions for every model version — and the archive pages — are reproducible
# without re-simulating. data/predictions/ is git-ignored; this is its home.
#
# Usage: ./scripts/upload_predictions_release.sh [tag]
#   tag defaults to "prediction-archive"

set -euo pipefail

REPO="jackleh/MLB-Predictions"
TAG="${1:-prediction-archive}"
ASSET="predictions.tar.gz"
PRED_DIR="$(dirname "$0")/../data/predictions"

if [ ! -f "$PRED_DIR/versions.json" ]; then
  echo "No prediction store found at $PRED_DIR (run the app to generate one)." >&2
  exit 1
fi

echo "Bundling prediction archive..."
# Archive the *contents* of data/predictions/ so it restores into that dir.
tar -czf "$ASSET" -C "$PRED_DIR" .

echo "Asset size: $(du -sh "$ASSET" | cut -f1)"
echo "Versions included:"
python3 - "$PRED_DIR/versions.json" <<'PY'
import json, sys
for v in json.load(open(sys.argv[1])):
    print(f"  {v['version']}  feat v{v.get('feature_version','?')}  {v.get('n_games','?')} games  {v.get('created_at','')[:10]}")
PY

# Delete existing release/tag if present so we can re-upload
gh release delete "$TAG" --repo "$REPO" --yes 2>/dev/null || true

echo "Creating release $TAG and uploading $ASSET..."
gh release create "$TAG" "$ASSET" \
  --repo "$REPO" \
  --title "Prediction Archive" \
  --notes "Frozen, model-versioned predictions (data/predictions/) for every trained model version. Restored on first run via bootstrap_predictions_cache so the archive pages and historical accuracy work without re-simulating." \
  --prerelease

rm "$ASSET"
echo "Done. Run 'gh release download $TAG -R $REPO -D data/predictions/' to restore."
