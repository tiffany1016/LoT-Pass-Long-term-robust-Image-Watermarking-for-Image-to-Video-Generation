#!/usr/bin/env python3
"""Inventory official ChronoMagic-150 meteorological sources and extract frames."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "chronomagic150_global_appearance"
DATASET = EXP / "dataset"
REFERENCES = EXP / "reference_videos"
FRAMES = EXP / "source_frames/original"
CSV_URL = "https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench/resolve/main/Captions/ChronoMagic-Bench-150.csv?download=true"
ZIP_URL = "https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench/resolve/main/Videos/Reference_Videos.zip?download=true"
ZIP_PATH = DATASET / "Reference_Videos.zip"
RAW_CSV = DATASET / "ChronoMagic-Bench-150.csv"
INVENTORY_CSV = DATASET / "chronomagic150_meteorological_all.csv"
COUNTS_JSON = DATASET / "metadata_count_summary.json"
RETRIEVAL_JSON = DATASET / "reference_retrieval_status.json"
EXPECTED_ROWS = 150
CHUNK = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HttpRangeFile(io.RawIOBase):
    """Seekable HTTP Range reader so only requested ZIP members are transferred."""

    def __init__(self, url: str):
        self.url, self.position, self.cache = url, 0, {}
        response = requests.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.position = offset
        elif whence == os.SEEK_CUR:
            self.position += offset
        elif whence == os.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        chunks = []
        while size:
            start = (self.position // CHUNK) * CHUNK
            end = min(self.size, start + CHUNK)
            if start not in self.cache:
                response = requests.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end - 1}"},
                    timeout=(30, 120),
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"Server ignored range request (HTTP {response.status_code})")
                self.cache = {start: response.content}
            data = self.cache[start]
            offset = self.position - start
            length = min(size, len(data) - offset)
            if length <= 0:
                raise IOError("Empty data in HTTP range response")
            chunks.append(data[offset : offset + length])
            self.position += length
            size -= length
        return b"".join(chunks)


def download_small_metadata() -> None:
    DATASET.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.is_file():
        response = requests.get(CSV_URL, timeout=(30, 120))
        response.raise_for_status()
        RAW_CSV.write_bytes(response.content)
    expected = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    print(f"Official metadata: {RAW_CSV.relative_to(ROOT)} ({RAW_CSV.stat().st_size:,} bytes; SHA-256 {expected})")


def load_metadata() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with RAW_CSV.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} ChronoMagic-150 records, found {len(rows)}")
    required = {"videoid", "name", "sub_category", "main_category"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Unexpected metadata columns: {list(rows[0]) if rows else []}")
    ids = [row["videoid"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate videoid values in official metadata")
    meteorological = [row.copy() for row in rows if row["main_category"].strip().casefold() == "meteorological"]
    return rows, meteorological


def write_inventory(rows: list[dict[str, str]], meteorological: list[dict[str, str]]) -> None:
    url = "https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench/tree/main/Videos/Reference_Videos.zip"
    fields = ["sample_id", "main_category", "sub_category", "official_caption", "reference_video_url",
              "reference_video_member", "reference_video_path", "reference_asset_status", "project_label",
              "global_appearance_candidate", "selection_reason", "exclusion_reason", "source_frame_original",
              "frame_timestamp_seconds", "initial_state_match", "candidate_group"]
    output = []
    for row in meteorological:
        output.append({
            "sample_id": row["videoid"], "main_category": row["main_category"],
            "sub_category": row["sub_category"], "official_caption": row["name"],
            "reference_video_url": url, "reference_video_member": f"{row['videoid']}.mp4",
            "reference_video_path": "", "reference_asset_status": "pending",
            "project_label": "", "global_appearance_candidate": "uncertain",
            "selection_reason": "", "exclusion_reason": "", "source_frame_original": "",
            "frame_timestamp_seconds": "0.0", "initial_state_match": "uncertain", "candidate_group": "",
        })
    with INVENTORY_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    counts = {
        "dataset": "BestWishYsh/ChronoMagic-Bench, ChronoMagic-Bench-150",
        "source_csv": "Captions/ChronoMagic-Bench-150.csv",
        "source_csv_sha256": sha256(RAW_CSV),
        "total_records": len(rows),
        "main_category_counts_preserving_blank": dict(Counter(r["main_category"] for r in rows)),
        "normalized_meteorological_count": len(meteorological),
        "meteorological_subcategory_counts": dict(Counter(r["sub_category"] for r in meteorological)),
        "expected_reference_video_archive": "Videos/Reference_Videos.zip",
        "official_metadata_commit": "d319674ef8d40d8e9a3021894ab2b30809f4f705",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    COUNTS_JSON.write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ChronoMagic-150: {len(rows)} records; meteorological: {len(meteorological)}")
    print("Main categories:", counts["main_category_counts_preserving_blank"])
    print("Meteorological subcategories:", counts["meteorological_subcategory_counts"])


def retrieve_meteorological_videos(meteorological: list[dict[str, str]]) -> None:
    REFERENCES.mkdir(parents=True, exist_ok=True)
    video_root = HttpRangeFile(ZIP_URL)
    archive = zipfile.ZipFile(video_root)
    available = {info.filename: info for info in archive.infolist()}
    missing = [row["videoid"] for row in meteorological if f"{row['videoid']}.mp4" not in available]
    if missing:
        raise RuntimeError(f"Official reference archive lacks {len(missing)} meteorological IDs: {missing}")
    status = {"archive_url": ZIP_URL.split("?", 1)[0], "archive_total_size": video_root.size,
              "archive_sha256": "not downloaded as a whole; official ZIP central directory and member CRCs verified",
              "zip_member_count": len(available), "expected_metadata_members_found": len(meteorological),
              "retrieved": [], "created_at_utc": datetime.now(timezone.utc).isoformat()}
    for index, row in enumerate(meteorological, start=1):
        video_id = row["videoid"]
        member = f"{video_id}.mp4"
        output = REFERENCES / member
        if not output.is_file() or output.stat().st_size != available[member].file_size:
            partial = output.with_suffix(".mp4.partial")
            with archive.open(member) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if partial.stat().st_size != available[member].file_size:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Extracted size mismatch for {member}")
            partial.replace(output)
        status["retrieved"].append({
            "sample_id": video_id, "member": member, "path": str(output.relative_to(ROOT)),
            "size_bytes": output.stat().st_size, "sha256": sha256(output), "zip_crc32": f"{available[member].CRC:08x}",
            "status": "available_and_extracted",
        })
        print(f"Reference asset {index}/{len(meteorological)}: {video_id}", flush=True)
    RETRIEVAL_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive.close()


def extract_first_frames(meteorological: list[dict[str, str]]) -> None:
    import av
    from PIL import Image

    FRAMES.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for index, row in enumerate(meteorological, start=1):
        sample_id = row["videoid"]
        video_path = REFERENCES / f"{sample_id}.mp4"
        image_path = FRAMES / f"{sample_id}.png"
        if not image_path.is_file():
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                frame = next(container.decode(stream), None)
                if frame is None:
                    raise RuntimeError(f"Video has no decodable first frame: {sample_id}")
                image = frame.to_image().convert("RGB")
                image.save(image_path, format="PNG", optimize=True)
                timestamp = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
                dimensions = image.size
        else:
            with Image.open(image_path) as image:
                dimensions = image.size
            timestamp = 0.0
        rows_out.append({"sample_id": sample_id, "path": str(image_path.relative_to(ROOT)),
                         "timestamp_seconds": timestamp, "width": dimensions[0], "height": dimensions[1]})
        print(f"First frame {index}/{len(meteorological)}: {sample_id}", flush=True)
    (DATASET / "meteorological_first_frames.json").write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    download_small_metadata()
    rows, meteorological = load_metadata()
    write_inventory(rows, meteorological)
    retrieve_meteorological_videos(meteorological)
    extract_first_frames(meteorological)


if __name__ == "__main__":
    main()
