#!/usr/bin/env python3
"""Resumable, frame-grouped ARPIL tile builder.

Each successfully processed source frame gets its own manifest fragment in
``output-dir/frame_manifests``.  Re-running the command skips those fragments,
so a Colab interruption never causes completed frames to be downloaded or tiled
again.  The combined manifest uses a frame-level validation split, preventing
patches from the same solar observation from leaking between train and val.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

# Reuse the tested upstream-download and tile helper functions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_arpil_3ch_tiles import (  # noqa: E402
    DEFAULT_CHANNELS,
    MASK_SPLIT_URLS,
    core_key_from_mask_path,
    download_core_s3,
    download_mask_archive,
    extract_mask,
    index_mask_archive,
    iter_starts,
    load_core_channels,
    load_mask,
    read_csv_rows,
    save_compressed,
)

import boto3
from botocore import UNSIGNED
from botocore.client import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ARPIL tiles incrementally without reprocessing completed frames.")
    parser.add_argument("--output-dir", required=True, help="Persistent output directory, preferably on Google Drive")
    parser.add_argument("--split", choices=sorted(MASK_SPLIT_URLS), default="train", help="Official ARPIL split to download")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--max-frames", type=int, default=24, help="New source frames to process in this invocation")
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many unprocessed official rows")
    parser.add_argument(
        "--sampling", choices=["random", "sequential"], default="random",
        help="Choose diverse random official frames (default) or chronological rows",
    )
    parser.add_argument(
        "--selection-file", default=None,
        help="Optional JSON file that freezes the selected source-frame IDs across Colab sessions",
    )
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--min-mask-fraction", type=float, default=0.0025)
    parser.add_argument("--keep-empty-every", type=int, default=32)
    parser.add_argument("--mask-key", default="union_with_intersect")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-free-disk-gb", type=float, default=15.0,
        help="Stop safely before processing another frame when local free disk falls below this reserve (default: 15 GB)",
    )
    return parser


def frame_id(row: dict[str, str]) -> str:
    return Path(row["file_path"]).stem


def load_fragment_rows(fragment_dir: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for path in sorted(fragment_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            result[path.stem] = list(csv.DictReader(handle))
    return result


def write_fragment(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_combined_manifest(
    output_dir: Path,
    fragments: dict[str, list[dict[str, str]]],
    fieldnames: list[str],
    val_ratio: float,
    seed: int,
) -> None:
    frame_ids = sorted(frame for frame, rows in fragments.items() if rows)
    shuffled = frame_ids[:]
    random.Random(seed).shuffle(shuffled)
    val_frames = set(shuffled[: max(1, int(len(shuffled) * val_ratio))]) if len(shuffled) > 1 else set()

    rows: list[dict[str, str]] = []
    for source_frame in frame_ids:
        split = "val" if source_frame in val_frames else "train"
        for row in fragments[source_frame]:
            row = dict(row)
            row["split"] = split
            rows.append(row)

    manifest = output_dir / "manifest.csv"
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)

    state = {
        "completed_frames": len(frame_ids),
        "tiles": len(rows),
        "train_frames": len(frame_ids) - len(val_frames),
        "val_frames": len(val_frames),
        "manifest": str(manifest),
    }
    (output_dir / "progress.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"Completed source frames: {state['completed_frames']}")
    print(f"Tiles in combined manifest: {state['tiles']}")
    print(f"Frame-grouped split: {state['train_frames']} train frames | {state['val_frames']} val frames")
    print(f"Manifest written: {manifest}")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    mask_root = output_dir / "masks"
    fragment_dir = output_dir / "frame_manifests"
    for directory in (image_root, mask_root, fragment_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for channel in args.channels:
        (image_root / channel).mkdir(parents=True, exist_ok=True)

    fieldnames = ["sample_id", "split", *(f"image_{channel}" for channel in args.channels), "mask"]
    fragments = load_fragment_rows(fragment_dir)
    completed = set(fragments)

    rows = read_csv_rows(MASK_SPLIT_URLS[args.split])
    rows = [row for row in rows if row.get("present", "0") not in {"0", "0.0", "", None} and row.get("file_path")]
    rows_by_id = {frame_id(row): row for row in rows}
    if args.selection_file:
        selection_path = Path(args.selection_file)
        if selection_path.exists():
            selected_ids = json.loads(selection_path.read_text(encoding="utf-8"))["frame_ids"]
            selected_rows = [rows_by_id[identifier] for identifier in selected_ids if identifier in rows_by_id]
            print(f"Using frozen selection: {selection_path} ({len(selected_rows)} source frames)")
        else:
            selected_rows = list(rows)
            if args.sampling == "random":
                random.Random(args.seed).shuffle(selected_rows)
            selected_rows = selected_rows[args.start_index : args.start_index + args.max_frames]
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = selection_path.with_suffix(selection_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps({"split": args.split, "seed": args.seed, "frame_ids": [frame_id(row) for row in selected_rows]}, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(selection_path)
            print(f"Created frozen selection: {selection_path} ({len(selected_rows)} source frames)")
        pending = [row for row in selected_rows if frame_id(row) not in completed]
    else:
        pending = [row for row in rows if frame_id(row) not in completed]
        if args.sampling == "random":
            random.Random(args.seed).shuffle(pending)
        pending = pending[args.start_index : args.start_index + args.max_frames]

    free_gb = shutil.disk_usage(output_dir).free / (1024 ** 3)
    print(f"Local free disk: {free_gb:.1f} GB (reserve: {args.min_free_disk_gb:.1f} GB)")
    print(f"Frame sampling: {args.sampling}")
    print(f"Previously completed frames: {len(completed)}")
    print(f"New frames selected this run: {len(pending)}")
    if free_gb < args.min_free_disk_gb:
        raise SystemExit(
            f"Only {free_gb:.1f} GB is free; refusing to start below the "
            f"{args.min_free_disk_gb:.1f} GB safety reserve. Free space or lower --min-free-disk-gb deliberately."
        )
    if not pending:
        write_combined_manifest(output_dir, fragments, fieldnames, args.val_ratio, args.seed)
        print("No new frames remain for this selection; nothing was downloaded twice.")
        return

    archive_path = output_dir / "download_cache" / "data.tar.gz"
    download_mask_archive(archive_path)
    members = index_mask_archive(archive_path)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    empty_counter = 0

    with tempfile.TemporaryDirectory(prefix="arpil_resume_") as tmp:
        temporary_dir = Path(tmp)
        for index, row in enumerate(pending, start=1):
            free_gb = shutil.disk_usage(output_dir).free / (1024 ** 3)
            if free_gb < args.min_free_disk_gb:
                print(
                    f"Stopping safely: {free_gb:.1f} GB free is below the "
                    f"{args.min_free_disk_gb:.1f} GB reserve. Completed frames remain saved."
                )
                write_combined_manifest(output_dir, fragments, fieldnames, args.val_ratio, args.seed)
                return
            timestamp = frame_id(row)
            core_key = core_key_from_mask_path(row["file_path"])
            nc_path = temporary_dir / f"{timestamp}.nc"
            h5_path = temporary_dir / f"{timestamp}.h5"
            print(f"[{index}/{len(pending)}] downloading core={core_key} mask={row['file_path']}")
            download_core_s3(s3_client, core_key, nc_path)
            extract_mask(archive_path, members, row["file_path"], h5_path)

            image_stack = load_core_channels(nc_path, args.channels)
            mask = load_mask(h5_path, args.mask_key)
            if image_stack.shape[1:] != mask.shape:
                raise ValueError(f"Shape mismatch for {timestamp}: image={image_stack.shape[1:]}, mask={mask.shape}")

            frame_rows: list[dict[str, str]] = []
            for y in iter_starts(mask.shape[0], args.patch_size, args.stride):
                for x in iter_starts(mask.shape[1], args.patch_size, args.stride):
                    mask_patch = mask[y : y + args.patch_size, x : x + args.patch_size]
                    if float(mask_patch.mean()) < args.min_mask_fraction:
                        empty_counter += 1
                        if args.keep_empty_every <= 0 or empty_counter % args.keep_empty_every != 0:
                            continue
                    sample_id = f"{timestamp}_y{y:04d}_x{x:04d}"
                    record = {"sample_id": sample_id, "split": "train"}
                    for channel_index, channel in enumerate(args.channels):
                        path = image_root / channel / f"{sample_id}.npz"
                        save_compressed(path, image_stack[channel_index, y : y + args.patch_size, x : x + args.patch_size])
                        record[f"image_{channel}"] = os.path.relpath(path, start=output_dir).replace(os.sep, "/")
                    mask_path = mask_root / f"{sample_id}.npz"
                    save_compressed(mask_path, mask_patch)
                    record["mask"] = os.path.relpath(mask_path, start=output_dir).replace(os.sep, "/")
                    frame_rows.append(record)

            # Atomic per-frame commit: future invocations skip it only after all
            # of its data and its manifest fragment have been written.
            write_fragment(fragment_dir / f"{timestamp}.csv", frame_rows, fieldnames)
            fragments[timestamp] = frame_rows
            nc_path.unlink(missing_ok=True)
            h5_path.unlink(missing_ok=True)
            write_combined_manifest(output_dir, fragments, fieldnames, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
