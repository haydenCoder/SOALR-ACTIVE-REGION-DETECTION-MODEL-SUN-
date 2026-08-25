#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/raw}"
ZENODO_URL="https://zenodo.org/records/7950721/files/MLMT-CNN_Solar-Active-Regions_Bounding-Box_and_Segmentation_Annotations_for_Deep-Learning_Application.zip?download=1"
ARCHIVE_NAME="MLMT-CNN_Solar-Active-Regions_Bounding-Box_and_Segmentation_Annotations_for_Deep-Learning_Application.zip"
ARCHIVE_PATH="${RAW_DIR}/${ARCHIVE_NAME}"
EXTRACT_ROOT="${RAW_DIR}/zenodo_extract"
UAD_ROOT="${RAW_DIR}/Solar_data_UAD"
TOOLS_DIR="${RAW_DIR}/tools"
NODE_TOOLS_DIR="${TOOLS_DIR}/npm-7zip"
SEVEN_Z_BIN="${NODE_TOOLS_DIR}/node_modules/7zip-bin-full/linux/x64/7zz"

ensure_7zz() {
  # Ubuntu's p7zip-full is version 16.02 in Colab and cannot read this
  # dataset's RAR compression method. Prefer a current official 7-Zip binary.
  if command -v 7zz >/dev/null 2>&1; then
    echo "7zz"
    return
  fi
  if [[ -x "${SEVEN_Z_BIN}" ]]; then
    echo "${SEVEN_Z_BIN}"
    return
  fi
  if command -v npm >/dev/null 2>&1; then
    mkdir -p "${NODE_TOOLS_DIR}"
    echo "Installing current standalone 7-Zip binary..." >&2
    (
      cd "${NODE_TOOLS_DIR}"
      if [[ ! -f package.json ]]; then
        npm init -y >/dev/null 2>&1
      fi
      npm install 7zip-bin-full --no-save >/dev/null
    )
    chmod +x "${SEVEN_Z_BIN}"
    echo "${SEVEN_Z_BIN}"
    return
  fi
  echo "A current 7-Zip '7zz' binary is required; npm was not found to install it." >&2
  exit 1
}

extract_archive() {
  local archive="$1"
  local destination="$2"
  local seven_zip_cmd

  mkdir -p "${destination}"
  seven_zip_cmd="$(ensure_7zz)"
  "${seven_zip_cmd}" x -y -o"${destination}" "${archive}" >/dev/null
}

mkdir -p "${RAW_DIR}" "${EXTRACT_ROOT}" "${UAD_ROOT}"

echo "[1/5] Downloading dataset archive from Zenodo..."
if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  wget -O "${ARCHIVE_PATH}" "${ZENODO_URL}"
else
  echo "Archive already exists at ${ARCHIVE_PATH}; skipping download."
fi

echo "[2/5] Extracting outer ZIP with 7-Zip..."
extract_archive "${ARCHIVE_PATH}" "${EXTRACT_ROOT}"

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

echo "[3/5] Extracting UAD channel archives and masks with 7-Zip..."
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
