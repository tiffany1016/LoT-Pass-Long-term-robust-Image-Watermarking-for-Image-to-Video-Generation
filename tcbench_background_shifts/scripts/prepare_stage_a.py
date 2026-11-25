#!/usr/bin/env python3
"""Census official TC-Bench background-shifts entries and retrieve their references."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "tcbench_background_shifts"
SOURCE = EXP / "source_repo"
METADATA = EXP / "metadata"
REFERENCES = EXP / "references"
EXPECTED_COUNT = 18
TARGET_SIZE = (720, 480)  # width, height; keep consistent with CogVideoX-I2V.
TARGET_RATIO = TARGET_SIZE[0] / TARGET_SIZE[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def video_id(url: str) -> str:
    parsed = urlparse(url)
    value = parse_qs(parsed.query).get("v", [None])[0]
    if not value and parsed.hostname in {"youtu.be", "www.youtu.be"}:
        value = parsed.path.strip("/").split("/")[0]
    if not value:
        raise ValueError(f"Cannot extract a YouTube video ID from {url!r}")
    return value


def parse_record(record: dict) -> dict:
    official_id = int(record["id"])
    prompt = record.get("prompt")
    video = record.get("video")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"TC{official_id}: missing nonempty prompt")
    if not isinstance(video, list) or len(video) < 2:
        raise ValueError(f"TC{official_id}: missing [URL, start,end] video reference")
    url = str(video[0]).strip()
    pieces = str(video[1]).split(",")
    if len(pieces) != 2:
        raise ValueError(f"TC{official_id}: malformed reference interval {video[1]!r}")
    start, end = map(float, pieces)
    if not url.startswith(("https://www.youtube.com/watch", "https://youtu.be/")) or start < 0 or end <= start:
        raise ValueError(f"TC{official_id}: invalid reference URL or interval")
    return {
        "case_id": f"TC{official_id}", "official_id": official_id,
        "prompt_official": prompt, "reference_video_url": url,
        "reference_video_id": video_id(url), "reference_start_sec": start,
        "reference_end_sec": end, "keywords": record.get("keywords", []),
        "original_record": record,
    }


def prepare_census() -> tuple[list[dict], dict]:
    metadata_path = SOURCE / "i2v_prompts.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Official TC-Bench metadata not found: {metadata_path}")
    source_rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    category = [r for r in source_rows if r.get("type") == "background shifts"]
    rows = [parse_record(record) for record in category]
    rows.sort(key=lambda row: row["official_id"])
    if len(rows) != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} official entries; pinned metadata has {len(rows)}")
    ids = [r["official_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate official IDs in the background-shifts category")

    METADATA.mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata_hash = sha256(metadata_path)
    snapshot = [r["original_record"] for r in rows]
    (METADATA / "official_background_shifts_all.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_rows = [{**r, "keywords": json.dumps(r["keywords"], ensure_ascii=False),
                 "original_record_json": json.dumps(r["original_record"], ensure_ascii=False, sort_keys=True)} for r in rows]
    common_fields = ["case_id", "official_id", "prompt_official", "reference_video_url", "reference_video_id",
                     "reference_start_sec", "reference_end_sec", "keywords", "original_record_json"]
    write_csv(METADATA / "official_background_shifts_all.csv", csv_rows, common_fields)

    initial_rows = []
    for row in rows:
        initial_rows.append({
            **row, "source_image_original_path": "", "source_image_model_ready_path": "",
            "reference_clip_path": "", "download_status": "pending",
            "input_matches_initial_state": "uncertain", "expected_change_mechanism": "uncertain",
            "expected_whole_frame_change": "uncertain", "possible_camera_motion": "unknown",
        })
    initial_fields = common_fields + ["source_image_original_path", "source_image_model_ready_path", "reference_clip_path",
                                     "download_status", "input_matches_initial_state", "expected_change_mechanism",
                                     "expected_whole_frame_change", "possible_camera_motion"]
    write_csv(METADATA / "all_18_initial.csv", initial_rows, initial_fields)

    refs: dict[str, list[dict]] = {}
    for row in rows:
        refs.setdefault(row["reference_video_id"], []).append({
            "case_id": row["case_id"], "url": row["reference_video_url"],
            "start_sec": row["reference_start_sec"], "end_sec": row["reference_end_sec"],
        })
    manifest = {
        "study_name": "TC-Bench-derived first-frame I2V evaluation of all official background-shifts entries",
        "official_category": "background shifts", "official_entry_count": len(rows),
        "expected_count": EXPECTED_COUNT, "official_ids": ids,
        "source_repository": "https://github.com/weixi-feng/TC-Bench",
        "source_repository_commit": source_commit,
        "metadata_file": str(metadata_path.relative_to(ROOT)), "metadata_sha256": metadata_hash,
        "unique_reference_video_count": len(refs), "reference_video_provenance": refs,
        "selection_policy": "census of all exact-type entries; no sampling or eligibility filtering",
        "preprocessing_policy": {
            "model_input_size_wh": list(TARGET_SIZE),
            "policy": "center crop to 3:2, then Pillow LANCZOS resize to 720x480; no stretch or color correction",
        },
        "reference_status": "pending retrieval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(METADATA / "dataset_census_manifest.json", manifest)
    print(f"Censused {len(rows)} entries (IDs {ids[0]}–{ids[-1]}); {len(refs)} unique YouTube sources.")
    print(f"Pinned TC-Bench commit: {source_commit}; metadata SHA-256: {metadata_hash}")
    return rows, manifest


def center_crop_resize(image):
    from PIL import Image
    image = image.convert("RGB")
    width, height = image.size
    if width / height > TARGET_RATIO:
        crop_width = int(height * TARGET_RATIO)
        left = (width - crop_width) // 2
        box = [left, 0, crop_width, height]
        axis = "horizontal"
    else:
        crop_height = int(width / TARGET_RATIO)
        top = (height - crop_height) // 2
        box = [0, top, width, crop_height]
        axis = "vertical"
    ready = image.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
    ready = ready.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    return ready, {"original_size_wh": [width, height], "crop_box_xywh": box, "crop_axis": axis,
                   "target_size_wh": list(TARGET_SIZE), "resampling": "Pillow LANCZOS"}


def retrieve_references(rows: list[dict]) -> list[dict]:
    import av
    from PIL import Image
    from yt_dlp import YoutubeDL

    original_dir = REFERENCES / "original_videos"
    clip_dir = REFERENCES / "reference_clips"
    frame_dir = REFERENCES / "source_frames_original"
    ready_dir = REFERENCES / "source_frames_model_ready"
    for directory in (original_dir, clip_dir, frame_dir, ready_dir):
        directory.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["reference_video_id"], []).append(row)
    source_status: dict[str, dict] = {}
    for source_id, items in sorted(grouped.items()):
        url = items[0]["reference_video_url"]
        existing = next((p for p in original_dir.glob(f"{source_id}.*") if p.is_file()
                         and not p.name.endswith((".part", ".ytdl"))), None)
        if existing is not None and existing.stat().st_size > 0:
            source_status[source_id] = {"status": "downloaded", "path": str(existing.relative_to(ROOT)),
                                        "sha256": sha256(existing), "url": url, "reused": True}
            continue
        outtmpl = str(original_dir / f"{source_id}.%(ext)s")
        options = {
            # TC-Bench references are used as visual input; video-only formats
            # avoid requiring a progressive A/V format that many current YT
            # uploads no longer expose. Cap downloads at 1080p for Stage A.
            "format": "bestvideo[height<=1080]/bestvideo/best",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "retries": 2,
            "fragment_retries": 2,
            "continuedl": True,
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
            "overwrites": False,
        }
        try:
            try:
                with YoutubeDL(options) as ydl:
                    ydl.extract_info(url, download=True)
            except Exception as first_error:
                # Some videos are exposed only through YouTube's Android
                # player client in this runtime. Keep the same source URL and
                # format cap, and record failure only if both clients fail.
                android_options = {
                    **options,
                    "extractor_args": {"youtube": {"player_client": ["android"]}},
                }
                try:
                    with YoutubeDL(android_options) as ydl:
                        ydl.extract_info(url, download=True)
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"default client failed ({type(first_error).__name__}: {first_error}); "
                        f"Android client failed ({type(fallback_error).__name__}: {fallback_error})"
                    ) from fallback_error
            downloaded = next((p for p in original_dir.glob(f"{source_id}.*")
                               if p.is_file() and not p.name.endswith((".part", ".ytdl"))), None)
            if downloaded is None or downloaded.stat().st_size == 0:
                raise RuntimeError("yt-dlp finished without producing a nonempty source file")
            source_status[source_id] = {"status": "downloaded", "path": str(downloaded.relative_to(ROOT)),
                                        "sha256": sha256(downloaded), "url": url, "reused": False}
        except Exception as exc:
            source_status[source_id] = {"status": "source_unavailable", "url": url,
                                        "error_type": type(exc).__name__, "error": str(exc)}
            print(f"[SOURCE ERROR] {source_id}: {type(exc).__name__}: {exc}", flush=True)

    rows_out = []
    extraction_errors = []
    for row in rows:
        source_id = row["reference_video_id"]
        source = source_status[source_id]
        result = {**row, "download_status": source["status"], "source_image_original_path": "",
                  "source_image_model_ready_path": "", "reference_clip_path": "",
                  "reference_source_video_path": source.get("path", ""),
                  "error": source.get("error", ""),
                  "input_matches_initial_state": "uncertain", "expected_change_mechanism": "uncertain",
                  "expected_whole_frame_change": "uncertain", "possible_camera_motion": "unknown"}
        if source["status"] != "downloaded":
            rows_out.append(result)
            continue
        start, end = row["reference_start_sec"], row["reference_end_sec"]
        frame_path = frame_dir / f"{row['case_id']}.png"
        ready_path = ready_dir / f"{row['case_id']}.png"
        try:
            with av.open(str(ROOT / source["path"])) as container:
                stream = container.streams.video[0]
                container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
                selected = None
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    source_time = float(frame.pts * stream.time_base)
                    if source_time < start:
                        continue
                    if source_time >= end:
                        break
                    selected = (frame, source_time)
                    break
                if selected is None:
                    raise RuntimeError(f"No decoded frame within official interval [{start}, {end}) seconds")
                frame, source_time = selected
                first_image = frame.to_image().convert("RGB")
                first_image.save(frame_path, format="PNG")
                local_time = source_time - start
            ready_image, transform = center_crop_resize(first_image)
            ready_image.save(ready_path, format="PNG", optimize=True)
            result.update({
                "download_status": "downloaded", "source_image_original_path": str(frame_path.relative_to(ROOT)),
                "source_image_model_ready_path": str(ready_path.relative_to(ROOT)),
                "reference_source_video_path": source["path"],
                "source_frame_time_original_seconds": source_time,
                "reference_clip_local_first_frame_seconds": local_time,
                "reference_source_video_sha256": source["sha256"],
                "source_frame_sha256": sha256(frame_path), "model_ready_sha256": sha256(ready_path),
                "source_image_original_size_wh": list(first_image.size),
                "model_ready_transform": transform,
                "source_frame_extraction_mode": "full source video: backward keyframe seek, then first PTS >= official start; no offset applied twice",
            })
        except Exception as exc:
            result["download_status"] = "frame_extraction_failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            extraction_errors.append({"case_id": row["case_id"], "error": result["error"]})
            print(f"[FRAME ERROR] {row['case_id']}: {type(exc).__name__}: {exc}", flush=True)
        rows_out.append(result)
        print(f"[REFERENCE {row['official_id']}] {result['download_status']}: {row['case_id']}", flush=True)

    atomic_json(METADATA / "reference_retrieval_status.json", {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "unique_source_status": source_status,
        "frame_extraction_errors": extraction_errors,
        "official_entries": len(rows_out),
        "downloaded_entries": sum(r["download_status"] == "downloaded" for r in rows_out),
        "missing_or_failed_entries": sum(r["download_status"] != "downloaded" for r in rows_out),
        "method": "yt-dlp downloads only official referenced sources once per unique YouTube video, falling back from the default client to the Android player client when needed; PyAV seeks backward to a keyframe and decodes the first frame with PTS >= official start within [start,end); frame is not re-encoded",
    })
    initial_fields = ["case_id", "official_id", "prompt_official", "reference_video_url", "reference_video_id",
                      "reference_start_sec", "reference_end_sec", "keywords", "original_record_json",
                      "reference_clip_path", "reference_source_video_path", "source_image_original_path", "source_image_model_ready_path",
                      "source_frame_time_original_seconds", "reference_clip_local_first_frame_seconds",
                      "reference_clip_sha256", "reference_source_video_sha256", "source_frame_sha256", "model_ready_sha256",
                      "source_image_original_size_wh", "model_ready_transform", "download_status", "error",
                      "source_frame_extraction_mode",
                      "input_matches_initial_state", "expected_change_mechanism", "expected_whole_frame_change",
                      "possible_camera_motion"]
    csv_ready = []
    for row in rows_out:
        item = {**row, "keywords": json.dumps(row["keywords"], ensure_ascii=False),
                "original_record_json": json.dumps(row["original_record"], ensure_ascii=False, sort_keys=True),
                "source_image_original_size_wh": json.dumps(row.get("source_image_original_size_wh", [])),
                "model_ready_transform": json.dumps(row.get("model_ready_transform", {}), ensure_ascii=False)}
        csv_ready.append(item)
    write_csv(METADATA / "all_18_initial.csv", csv_ready, initial_fields)
    print(f"References ready: {sum(r['download_status']=='downloaded' for r in rows_out)}/{len(rows_out)} entries.")


def main() -> None:
    if not (SOURCE / ".git").exists():
        raise FileNotFoundError(f"Clone the official TC-Bench repository into {SOURCE}")
    rows, _ = prepare_census()
    if "--census-only" not in sys.argv:
        retrieve_references(rows)


if __name__ == "__main__":
    main()
