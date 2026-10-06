#!/bin/bash
# Bundle the API (runtime modules + artifacts) into services/api/ and deploy it.
#   scripts/deploy_api.sh            -> vercel production deployment
#   scripts/deploy_api.sh --bundle   -> only bundle (e.g. for `docker build services/api`)
set -euo pipefail
cd "$(dirname "$0")/.."
API=services/api
rm -rf "$API/serve" "$API/artifacts"
mkdir -p "$API/serve" "$API/artifacts"
for m in __init__ encoder catalog github ranking explain app; do
  cp "src/firstpr/serve/$m.py" "$API/serve/"
done
cp -R data/serving/static "$API/artifacts/static"
mkdir -p "$API/artifacts/dynamic"
for f in issues.json.gz repo_features.json fresh.json.gz manifest.json; do
  cp "data/serving/dynamic/$f" "$API/artifacts/dynamic/"
done
du -sh "$API/artifacts"
if [[ "${1:-}" == "--bundle" ]]; then exit 0; fi
(cd "$API" && vercel deploy --prod --yes)
