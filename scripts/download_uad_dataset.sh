#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw}"
ZENODO_URL="https://zenodo.org/records/7950721/files/MLMT-CNN_Solar-Active-Regions_Bounding-Box_and_Segmentation_Annotations_for_Deep-Learning_Application.zip?download=1"
ARCHIVE_NAME="MLMT-CNN_Solar-Active-Regions_Bounding-Box_and_Segmentation_Annotations_for_Deep-Learning_Application.zip"
ARCHIVE_PATH="${RAW_DIR}/${ARCHIVE_NAME}"
EXTRACT_ROOT="${RAW_DIR}/zenodo_extract"
UAD_ROOT="${RAW_DIR}/Solar_data_UAD"

mkdir -p "${RAW_DIR}" "${EXTRACT_ROOT}" "${UAD_ROOT}"

echo "[1/5] Downloading dataset archive from Zenodo..."
if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  wget -O "${ARCHIVE_PATH}" "${ZENODO_URL}"
else
  echo "Archive already exists at ${ARCHIVE_PATH}; skipping download."
fi

echo "[2/5] Extracting outer ZIP..."
unzip -o "${ARCHIVE_PATH}" -d "${EXTRACT_ROOT}" >/dev/null

INNER_ROOT="$(find "${EXTRACT_ROOT}" -maxdepth 1 -type d -name 'MLMT-CNN_*' | head -n 1)"
if [[ -z "${INNER_ROOT}" ]]; then
  echo "Could not locate extracted dataset directory inside ${EXTRACT_ROOT}" >&2
  exit 1
fi

SOURCE_UAD="${INNER_ROOT}/Solar_data_UAD"
if [[ ! -d "${SOURCE_UAD}" ]]; then
  echo "Could not locate Solar_data_UAD inside ${INNER_ROOT}" >&2
  exit 1
fi

TOOLS_DIR="${RAW_DIR}/tools"
NODE_TOOLS_DIR="${TOOLS_DIR}/npm-7zip"
SEVEN_Z_BIN="${NODE_TOOLS_DIR}/node_modules/7zip-bin-full/linux/x64/7zz"

ensure_7zz() {
  if command -v 7z >/dev/null 2>&1; then
    echo "7z"
    return
  fi
  if command -v 7zz >/dev/null 2>&1; then
    echo "7zz"
    return
  fi
  if [[ -x "${SEVEN_Z_BIN}" ]]; then
    echo "${SEVEN_Z_BIN}"
    return
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required to bootstrap a local 7-Zip binary when 7z is unavailable." >&2
    exit 1
  fi

  mkdir -p "${NODE_TOOLS_DIR}"
  echo "Installing standalone 7-Zip binary via npm..."
  (
    cd "${NODE_TOOLS_DIR}"
    if [[ ! -f package.json ]]; then
      npm init -y >/dev/null 2>&1
    fi
    npm install 7zip-bin-full --no-save >/dev/null
  )
  chmod +x "${SEVEN_Z_BIN}"
  echo "${SEVEN_Z_BIN}"
}

extract_archive() {
  local archive="$1"
  local destination="$2"
  mkdir -p "${destination}"
  local lower="${archive,,}"

  if [[ "${lower}" == *.zip ]]; then
    unzip -o "${archive}" -d "${destination}" >/dev/null
    return
  fi

  local seven_zip_cmd
  seven_zip_cmd="$(ensure_7zz)"
  if [[ -n "${seven_zip_cmd}" ]]; then
    if "${seven_zip_cmd}" x -y -o"${destination}" "${archive}" >/dev/null; then
      return
    fi
  fi

  if command -v unrar >/dev/null 2>&1; then
    unrar x -o+ "${archive}" "${destination}/" >/dev/null
    return
  fi

  if command -v unrar-free >/dev/null 2>&1; then
    unrar-free -x "${archive}" "${destination}/" >/dev/null
    return
  fi

  echo "Failed to extract ${archive}. Install 7z or unrar." >&2
  exit 1
}

echo "[3/5] Extracting UAD channel archives and masks..."
extract_archive "${SOURCE_UAD}/training_images_171.rar" "${UAD_ROOT}/training_images_171"
extract_archive "${SOURCE_UAD}/training_images_195.rar" "${UAD_ROOT}/training_images_195"
extract_archive "${SOURCE_UAD}/training_images_284.rar" "${UAD_ROOT}/training_images_284"
extract_archive "${SOURCE_UAD}/training_images_304.rar" "${UAD_ROOT}/training_images_304"
extract_archive "${SOURCE_UAD}/Masks.zip" "${UAD_ROOT}/Masks"

if [[ -f "${SOURCE_UAD}/testing_images.rar" ]]; then
  echo "[4/5] Extracting optional test images archive..."
  extract_archive "${SOURCE_UAD}/testing_images.rar" "${UAD_ROOT}/testing_images"
else
  echo "[4/5] testing_images.rar not found; skipping optional extraction."
fi

echo "[5/5] Done. Extracted files live under ${UAD_ROOT}"
echo "Next: python scripts/prepare_uad_manifest.py --raw-root ${UAD_ROOT}"
