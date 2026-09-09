#!/usr/bin/env bash
# Extract Sylvest/libero_plus_rlds split zip from HF cache to ~/DATA/libero_plus_rlds/
#
# Usage:
#   bash b/script/libero/extract_libero_plus_rlds.sh
#   SNAP=/path/to/snapshot OUT_ROOT=~/DATA/libero_plus_rlds bash b/script/libero/extract_libero_plus_rlds.sh
set -euo pipefail

HF_SNAP="${SNAP:-/home/luogang/hf_home/hub/datasets--Sylvest--libero_plus_rlds/snapshots/fb0c7029b076030d5d57227229e4f7460def1f7c}"
OUT_ROOT="${OUT_ROOT:-/home/luogang/DATA/libero_plus_rlds}"
WORK_DIR="${WORK_DIR:-/home/luogang/DATA/.libero_plus_rlds_work}"
COMBINED_ZIP="${WORK_DIR}/libero_plus_mixdata_full.zip"
TFDS_DIR="${OUT_ROOT}/libero_mix/1.0.0"

log() { echo "[$(date '+%F %T')] $*"; }

ZIP_BIN="${ZIP_BIN:-}"
if [[ -z "${ZIP_BIN}" ]]; then
  for candidate in /home/luogang/miniforge3/bin/zip "$(command -v zip 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      ZIP_BIN="${candidate}"
      break
    fi
  done
fi
UNZIP_BIN="${UNZIP_BIN:-}"
if [[ -z "${UNZIP_BIN}" ]]; then
  for candidate in /home/luogang/miniforge3/bin/unzip "$(command -v unzip 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      UNZIP_BIN="${candidate}"
      break
    fi
  done
fi

[[ -d "${HF_SNAP}" ]] || { echo "HF snapshot not found: ${HF_SNAP}" >&2; exit 1; }
for part in libero_plus_mixdata.zip libero_plus_mixdata.z01 libero_plus_mixdata.z02; do
  [[ -e "${HF_SNAP}/${part}" ]] || { echo "Missing part: ${HF_SNAP}/${part}" >&2; exit 1; }
done

mkdir -p "${OUT_ROOT}" "${WORK_DIR}"

if [[ -f "${TFDS_DIR}/features.json" ]] && ls "${TFDS_DIR}"/libero_mix-train.tfrecord-* &>/dev/null; then
  log "Already extracted: ${TFDS_DIR}"
  exit 0
fi

cd "${HF_SNAP}"

if [[ ! -f "${COMBINED_ZIP}" ]]; then
  log "Combining split zip parts -> ${COMBINED_ZIP}"
  if [[ -n "${ZIP_BIN}" ]]; then
    "${ZIP_BIN}" -s 0 libero_plus_mixdata.zip --out "${COMBINED_ZIP}"
  else
    python3 - <<'PY' "${HF_SNAP}" "${COMBINED_ZIP}"
import sys
from pathlib import Path

snap = Path(sys.argv[1])
out = Path(sys.argv[2])
parts = [
    snap / "libero_plus_mixdata.zip",
    snap / "libero_plus_mixdata.z01",
    snap / "libero_plus_mixdata.z02",
]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("wb") as dst:
    for p in parts:
        print(f"append {p.name} ({p.stat().st_size / (1024**3):.2f} GiB)")
        with p.open("rb") as src:
            while True:
                chunk = src.read(64 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
print(f"combined -> {out}")
PY
  fi
else
  log "Reusing combined zip: ${COMBINED_ZIP}"
fi

STAGING="${WORK_DIR}/staging"
rm -rf "${STAGING}"
mkdir -p "${STAGING}"

log "Extracting archive (this may take 10-30+ minutes)..."
if [[ -n "${UNZIP_BIN}" ]]; then
  "${UNZIP_BIN}" -q "${COMBINED_ZIP}" -d "${STAGING}"
else
  python3 - <<'PY' "${COMBINED_ZIP}" "${STAGING}"
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
staging = Path(sys.argv[2])
staging.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(staging)
print(f"extracted to {staging}")
PY
fi

FOUND="$(find "${STAGING}" -type d -path '*/libero_mix/1.0.0' | head -1)"
[[ -n "${FOUND}" ]] || { echo "Could not find libero_mix/1.0.0 under ${STAGING}" >&2; exit 1; }

mkdir -p "$(dirname "${TFDS_DIR}")"
rm -rf "${TFDS_DIR}"
mv "${FOUND}" "${TFDS_DIR}"

log "TFDS data ready at: ${TFDS_DIR}"
log "Shard count: $(ls -1 "${TFDS_DIR}"/libero_mix-train.tfrecord-* 2>/dev/null | wc -l)"
log "Optional: rm -rf ${WORK_DIR}  # free combined zip + staging after verification"
