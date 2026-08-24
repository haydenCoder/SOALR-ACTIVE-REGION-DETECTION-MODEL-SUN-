"""Unified array loading for every file format used by this project.

The training pipeline has to read solar data that ships in a lot of different
containers:

===============  ==========================================================
Format           Where it comes from
===============  ==========================================================
``.h5``/``.hdf5``ARPIL / SuryaBench segmentation masks, plugin tiles
``.nc``          core-SDO frames (netCDF4)
``.fits``        SDO/HMI/AIA archives (SHARP, SMARP, SunPy samples)
``.npy``/``.npz``Pre-processed patches written by ``scripts/``
``.png``/...     The Zenodo UAD dataset
===============  ==========================================================

Every reader funnels through :func:`load_array` so new formats only have to be
registered in one place.

HDF5/netCDF files hold *named* datasets, so a plain path is not enough to know
what to read.  Two mechanisms select the dataset, both designed so the choice
can be changed later without touching code:

1. A per-path suffix in the manifest -- ``masks/frame.h5#union_with_intersect``
2. A pipeline-wide default key (``--hdf5-key`` on the CLI)

If neither resolves, the loader falls back to the first array-like dataset in
the file and searches nested groups recursively, so an unexpected layout
degrades to a warning instead of a crash.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LOGGER = logging.getLogger(__name__)

# ``.fits.gz`` etc. are handled by :func:`file_extension`, which understands
# compound suffixes.
HDF5_EXTENSIONS = {".h5", ".hdf5", ".he5", ".hdf"}
NETCDF_EXTENSIONS = {".nc", ".nc4", ".netcdf", ".cdf"}
FITS_EXTENSIONS = {".fits", ".fit", ".fts"}
NUMPY_EXTENSIONS = {".npy", ".npz"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SUPPORTED_EXTENSIONS = (
    HDF5_EXTENSIONS | NETCDF_EXTENSIONS | FITS_EXTENSIONS | NUMPY_EXTENSIONS | IMAGE_EXTENSIONS
)

#: Compression suffixes that may trail a real extension (``frame.fits.gz``).
_COMPRESSION_SUFFIXES = {".gz", ".bz2", ".xz", ".zst", ".z"}

#: Dataset names tried, in order, when no explicit key is supplied.
DEFAULT_HDF5_KEYS = (
    "union_with_intersect",
    "mask",
    "image",
    "data",
    "segmentation",
)


class ArrayLoadError(RuntimeError):
    """Raised when a file exists but cannot be turned into a 2-D array."""


def file_extension(path: Path) -> str:
    """Return the meaningful lowercase extension, ignoring compression suffixes.

    ``frame.fits.gz`` -> ``.fits`` so compressed archives route to the right
    reader instead of failing an extension check.
    """
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    while suffixes and suffixes[-1] in _COMPRESSION_SUFFIXES:
        suffixes.pop()
    return suffixes[-1] if suffixes else ""


def split_path_spec(spec: str | Path) -> tuple[Path, str | None]:
    """Split ``path/to/file.h5#dataset/name`` into its path and dataset key.

    ``#`` is not valid in the generated filenames, so this is unambiguous.  A
    spec without ``#`` returns ``(path, None)``.
    """
    text = str(spec)
    if "#" in text:
        raw_path, _, key = text.partition("#")
        key = key.strip()
        return Path(raw_path), key or None
    return Path(text), None


def is_supported(path: str | Path) -> bool:
    """Return ``True`` when :func:`load_array` has a reader for ``path``."""
    file_path, _ = split_path_spec(path)
    return file_extension(file_path) in SUPPORTED_EXTENSIONS


def squeeze_to_2d(array: np.ndarray, source: str = "<array>") -> np.ndarray:
    """Reduce ``array`` to two dimensions, preferring lossless squeezes.

    Solar data arrives with assorted leading axes: ``(1, H, W)`` for a
    single-frame cube, ``(H, W, 1)`` for a grayscale image, ``(T, H, W)`` for a
    time series.  Size-1 axes are dropped first (lossless).  Only if the result
    is still 3-D is the leading axis indexed, which is the conventional
    "take the first frame/band" behaviour.

    The previous implementation used ``array[..., 0]`` unconditionally, which
    silently turned a ``(1, H, W)`` cube into a ``(1, H)`` sliver.
    """
    array = np.asarray(array)
    if array.ndim > 2:
        # Drop only size-1 axes; this never discards real data.
        squeeze_axes = tuple(axis for axis, size in enumerate(array.shape) if size == 1)
        if squeeze_axes and array.ndim - len(squeeze_axes) >= 2:
            array = np.squeeze(array, axis=squeeze_axes)
    while array.ndim > 2:
        # Genuine extra axis (time or band): keep the first slice.
        LOGGER.debug("Reducing %s from shape %s by taking index 0 of axis 0", source, array.shape)
        array = array[0]
    if array.ndim != 2:
        raise ArrayLoadError(f"Expected a 2D image for {source}, got shape={array.shape}")
    return array


def _iter_hdf5_datasets(node: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Depth-first walk yielding ``(path, dataset)`` for every dataset in ``node``."""
    import h5py

    for name, child in node.items():
        child_path = f"{prefix}/{name}" if prefix else name
        if isinstance(child, h5py.Group):
            yield from _iter_hdf5_datasets(child, child_path)
        elif isinstance(child, h5py.Dataset):
            yield child_path, child


def resolve_hdf5_key(handle: Any, key: str | None, source: str) -> Any:
    """Locate the dataset to read inside an open HDF5 file.

    Resolution order:

    1. ``key`` as a literal path (supports nested ``group/dataset``)
    2. ``key`` matched against the *basename* of any nested dataset
    3. each name in :data:`DEFAULT_HDF5_KEYS`
    4. the first array-like dataset found anywhere in the file

    Returns the ``h5py.Dataset``.  Raises :class:`ArrayLoadError` if the file
    contains no datasets at all.
    """
    import h5py

    if key:
        node = handle.get(key)
        if isinstance(node, h5py.Dataset):
            return node
        # The key may name a dataset nested inside a group.
        basename = key.rsplit("/", 1)[-1]
        for dataset_path, dataset in _iter_hdf5_datasets(handle):
            if dataset_path == key or dataset_path.rsplit("/", 1)[-1] == basename:
                return dataset
        LOGGER.warning("HDF5 key %r not found in %s; falling back to auto-detection", key, source)

    datasets = list(_iter_hdf5_datasets(handle))
    if not datasets:
        raise ArrayLoadError(f"No HDF5 datasets found in {source}")

    by_basename = {path.rsplit("/", 1)[-1]: dataset for path, dataset in datasets}
    for candidate in DEFAULT_HDF5_KEYS:
        if candidate in by_basename:
            return by_basename[candidate]

    # Prefer a dataset that actually looks like an image over scalar metadata.
    for _, dataset in datasets:
        if getattr(dataset, "ndim", 0) >= 2:
            return dataset
    return datasets[0][1]


def hdf5_dataset_names(path: str | Path) -> list[str]:
    """List every dataset path in an HDF5 file. Useful for debugging a new layout."""
    import h5py

    file_path, _ = split_path_spec(path)
    with h5py.File(file_path, "r") as handle:
        return [name for name, _ in _iter_hdf5_datasets(handle)]


def _load_hdf5(path: Path, key: str | None) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ArrayLoadError(
            f"Reading {path.name} requires h5py. Install it with: pip install -r requirements.txt"
        ) from exc

    with h5py.File(path, "r") as handle:
        dataset = resolve_hdf5_key(handle, key, str(path))
        return np.asarray(dataset[()], dtype=np.float32)


def _load_netcdf(path: Path, key: str | None) -> np.ndarray:
    try:
        from netCDF4 import Dataset as NetCDFDataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ArrayLoadError(
            f"Reading {path.name} requires netCDF4. Install it with: pip install -r requirements.txt"
        ) from exc

    with NetCDFDataset(path, "r") as dataset:
        variables = list(dataset.variables)
        if key and key in dataset.variables:
            name = key
        else:
            if key:
                LOGGER.warning(
                    "netCDF variable %r not found in %s (available: %s); using the first 2D variable",
                    key,
                    path.name,
                    variables,
                )
            candidates = [n for n in variables if dataset.variables[n].ndim >= 2]
            if not candidates:
                raise ArrayLoadError(f"No 2D variables found in {path} (available: {variables})")
            name = candidates[0]
        # ``[:]`` yields a masked array; fill masked entries rather than let
        # them propagate as NaN through normalization.
        raw = dataset.variables[name][:]
        return np.asarray(np.ma.filled(raw, 0.0), dtype=np.float32)


def _load_fits(path: Path, key: str | None) -> np.ndarray:
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ArrayLoadError(
            f"Reading {path.name} requires astropy. Install it with: pip install -r requirements.txt"
        ) from exc

    with fits.open(path, memmap=False) as hdul:
        if key is not None:
            try:
                data = hdul[key].data  # HDU name or index
            except (KeyError, IndexError):
                LOGGER.warning("FITS HDU %r not found in %s; using the first image HDU", key, path.name)
                data = None
            if data is not None:
                return np.asarray(data, dtype=np.float32)
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) >= 2:
                return np.asarray(hdu.data, dtype=np.float32)
    raise ArrayLoadError(f"No image data found in FITS file {path}")


def _load_numpy(path: Path, key: str | None) -> np.ndarray:
    if file_extension(path) == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)

    with np.load(path, allow_pickle=False) as archive:
        names = list(archive.files)
        if not names:
            raise ArrayLoadError(f"Empty npz archive: {path}")
        if key and key in names:
            name = key
        elif len(names) == 1:
            name = names[0]
        else:
            # ``np.savez_compressed(path, arr)`` stores under 'arr_0'.
            name = "arr_0" if "arr_0" in names else names[0]
            LOGGER.debug("npz %s holds %s; using %r", path.name, names, name)
        return np.asarray(archive[name], dtype=np.float32)


def _load_image(path: Path, key: str | None) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        # "F" keeps 32-bit float precision for 16-bit TIFFs instead of
        # clipping them to 8-bit like "L" would.
        return np.asarray(image.convert("F"), dtype=np.float32)


def load_array(path: str | Path, key: str | None = None, *, as_2d: bool = True) -> np.ndarray:
    """Load any supported file into a ``float32`` numpy array.

    Parameters
    ----------
    path:
        File path, optionally with a ``#dataset`` suffix that overrides ``key``.
    key:
        Default dataset/variable/HDU name for container formats (HDF5, netCDF,
        FITS, npz). Ignored by plain image formats.
    as_2d:
        Squeeze the result to two dimensions (the default for segmentation).
        Pass ``False`` to keep multi-band cubes intact.
    """
    file_path, inline_key = split_path_spec(path)
    effective_key = inline_key or key

    if not file_path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {file_path}")

    extension = file_extension(file_path)
    if extension in HDF5_EXTENSIONS:
        array = _load_hdf5(file_path, effective_key)
    elif extension in NETCDF_EXTENSIONS:
        array = _load_netcdf(file_path, effective_key)
    elif extension in FITS_EXTENSIONS:
        array = _load_fits(file_path, effective_key)
    elif extension in NUMPY_EXTENSIONS:
        array = _load_numpy(file_path, effective_key)
    elif extension in IMAGE_EXTENSIONS:
        array = _load_image(file_path, effective_key)
    else:
        raise ArrayLoadError(
            f"Unsupported file extension {extension!r} for {file_path}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    array = np.asarray(array, dtype=np.float32)
    if as_2d:
        array = squeeze_to_2d(array, source=str(file_path))
    return array


def write_hdf5_tile(
    path: str | Path,
    image: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, Any] | None = None,
    *,
    image_key: str = "image",
    mask_key: str = "mask",
    compression: str | None = "gzip",
) -> Path:
    """Write one training tile as HDF5, atomically.

    Writing to a temporary file and renaming means an interrupted run can never
    leave a half-written ``.h5`` that later reads as corrupt.
    """
    import h5py

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_name(file_path.name + ".tmp")

    try:
        with h5py.File(temp_path, "w") as handle:
            handle.create_dataset(
                image_key, data=np.asarray(image, dtype=np.float32), compression=compression
            )
            handle.create_dataset(
                mask_key, data=np.asarray(mask, dtype=np.float32), compression=compression
            )
            for meta_key, value in (metadata or {}).items():
                handle.attrs[meta_key] = value
        temp_path.replace(file_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return file_path


def read_hdf5_tile(
    path: str | Path, *, image_key: str = "image", mask_key: str = "mask"
) -> tuple[np.ndarray, np.ndarray]:
    """Read an ``(image, mask)`` pair written by :func:`write_hdf5_tile`."""
    import h5py

    file_path, _ = split_path_spec(path)
    with h5py.File(file_path, "r") as handle:
        image = np.asarray(resolve_hdf5_key(handle, image_key, str(file_path))[()], dtype=np.float32)
        mask = np.asarray(resolve_hdf5_key(handle, mask_key, str(file_path))[()], dtype=np.float32)
    return image, mask
