#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/raw/real_nasa_samples}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${OUT_DIR}/sunpy" "${OUT_DIR}/sharp"

echo "Cloning SunPy data sample repository..."
git clone --depth 1 https://github.com/sunpy/data.git "${TMP_DIR}/sunpy-data" >/dev/null

cp -f "${TMP_DIR}/sunpy-data/sunpy/v1/AIA20110607_063302_0171_lowres.fits" "${OUT_DIR}/sunpy/"
cp -f "${TMP_DIR}/sunpy-data/sunpy/v1/AIA20110607_063307_0193_lowres.fits" "${OUT_DIR}/sunpy/"
cp -f "${TMP_DIR}/sunpy-data/sunpy/v1/HMI20110607_063211_los_lowres.fits" "${OUT_DIR}/sunpy/"

echo "Cloning SHARPs sample repository..."
git clone --depth 1 https://github.com/mbobra/SHARPs.git "${TMP_DIR}/SHARPs" >/dev/null

cp -f "${TMP_DIR}/SHARPs/files/hmi.sharp_cea_720s.377.20110215_020000_TAI.magnetogram.fits" "${OUT_DIR}/sharp/"
cp -f "${TMP_DIR}/SHARPs/files/hmi.sharp_cea_720s.377.20110215_020000_TAI.bitmap.fits" "${OUT_DIR}/sharp/"
cp -f "${TMP_DIR}/SHARPs/files/hmi.sharp_cea_720s.377.20110215_020000_TAI.continuum.fits" "${OUT_DIR}/sharp/"

echo "Downloaded real NASA/SDO sample FITS files to ${OUT_DIR}"
find "${OUT_DIR}" -maxdepth 2 -type f | sort
