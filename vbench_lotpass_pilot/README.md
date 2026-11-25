# VBench++ LoT-Pass camera-motion pilot

Stage A has prepared a deterministic 10-image, 40-prompt selection from the official VBench++ I2V image suite. Review `selection/approval_request.md` and `selection/contact_sheet.png`.

The selection package records all camera-motion prompts, eligibility checks, selected image provenance, exact official prompts, checksums, and canonical 720x480 preprocessing. Official original images, the official 8:5 crop variants, and canonical clean images are under `images/`.

**Stage B is gated:** do not encode watermarks, generate videos, decode frames, or run GPU analysis until explicit user approval of the current `selection/selection_manifest.json`.

The selection can be rebuilt with:

```bash
.venv/bin/python scripts/build_selection.py
```

That script downloads only the selected image files through byte-range requests to the official VBench++ Google Drive archives. It reads prompt metadata from the official VBench GitHub repository.

The portable results dashboard is `../vbench_lotpass_pilot_dashboard.html`. It embeds browser-compatible 480×320 H.264 video previews, so it does not need access to the sibling `runs/` directory for playback. Original generated videos remain unchanged at full resolution under `runs/`. Rebuild the dashboard with `uv run --python .venv/bin/python scripts/build_interactive_ba_acc_site.py`.
