#!/usr/bin/env bash
# Download test data, training data and trained models from the Google Drive
# links given in the PraNet and Polyp-PVT READMEs.
# Usage: bash scripts/download.sh <work_dir> [data_dir]
#   work_dir  zips and weights are cached here (can be on Google Drive)
#   data_dir  zips are extracted here; keep it on local disk, unzipping thousands
#             of small files onto the Drive mount is slow and can stop half-way
set -euo pipefail

WORK=${1:-/content/work}
DATA=${2:-/content/data}
mkdir -p "$WORK/downloads" "$WORK/weights" "$DATA"
cd "$WORK/downloads"

# PraNet README: TestDataset (5 sets), TrainDataset (Kvasir 900 + ClinicDB 550), PraNet-19.pth
[ -f TestDataset.zip ]  || gdown 1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao -O TestDataset.zip
[ -f TrainDataset.zip ] || gdown 1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb -O TrainDataset.zip
[ -f "$WORK/weights/PraNet-19.pth" ] || gdown 1lJv8XVStsp3oNKZHaSr42tawdMOq6FLP -O "$WORK/weights/PraNet-19.pth"

# Polyp-PVT README: trained model (4.7) and PVTv2-B2 backbone (4.3, Polyp-PVT's
# constructor loads it from ./pretrained_pth even at test time)
[ -d pvt_trained ]  || gdown --folder 1xC5Opwu5Afz4xiK5O9v4NnQOZY0A9-2j -O pvt_trained
[ -d pvt_backbone ] || gdown --folder 1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV -O pvt_backbone

find pvt_trained  -name '*.pth' -exec cp {} "$WORK/weights/PolypPVT.pth" \; -quit
find pvt_backbone -name 'pvt_v2_b2*.pth' -exec cp {} "$WORK/weights/pvt_v2_b2.pth" \; -quit

for z in TestDataset.zip TrainDataset.zip; do
  rm -rf "$DATA/${z%.zip}"
  unzip -q -o "$z" -d "$DATA"
done
# Remove macOS metadata that some of the archives ship with
rm -rf "$DATA/__MACOSX"

ls -lh "$WORK/weights"

# Fail loudly if a test set is incomplete (sizes from the PraNet README)
status=0
while read -r s n; do
  ni=$(ls "$DATA/TestDataset/$s/images" | wc -l)
  nm=$(ls "$DATA/TestDataset/$s/masks" | wc -l)
  printf '%-18s images=%4s masks=%4s expected=%4s\n' "$s" "$ni" "$nm" "$n"
  if [ "$ni" -ne "$n" ] || [ "$nm" -ne "$n" ]; then status=1; fi
done <<'EOF'
CVC-300 60
CVC-ClinicDB 62
Kvasir 100
CVC-ColonDB 380
ETIS-LaribPolypDB 196
EOF
if [ "$status" -ne 0 ]; then
  echo "Test data incomplete: delete $WORK/downloads/TestDataset.zip and run again." >&2
  exit 1
fi
