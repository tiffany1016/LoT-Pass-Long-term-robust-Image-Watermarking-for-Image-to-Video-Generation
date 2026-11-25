#!/usr/bin/env python3
"""Record the user's explicit approval of the ChronoMagic Stage A package."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "chronomagic150_global_appearance"
MANIFEST = EXP / "dataset/chronomagic150_generation_manifest.csv"
APPROVAL = EXP / "stage_a_review/approval_record.json"

digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
record = {
    "approved": True,
    "approved_at_utc": datetime.now(timezone.utc).isoformat(),
    "approved_by": "user in conversation",
    "generation_manifest_sha256": digest,
    "approval_action": "User instructed: you can start the experiment.",
}
APPROVAL.parent.mkdir(parents=True, exist_ok=True)
APPROVAL.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(f"Recorded approval for manifest SHA-256 {digest}")
