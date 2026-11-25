# VBench++ LoT-Pass pilot: Stage A selection for approval

This package contains a deterministic 10-image sample and 40 verbatim official camera-motion prompts. No watermark encoding, video generation, decoding, or GPU analysis has been run.

## Source and selection

- Official VBench++ prompt metadata: 1118 entries; 763 camera-motion entries across 109 images.
- Eligible images: 109 with all seven motion variants; unknown suffixes: 0; conflicting duplicates: 0.
- Selected categories: {'architecture': 3, 'indoor': 3, 'scenery': 3, 'abstract': 1}; selection seed: 42.
- Official images are from the VBench++ image suite. Canonical inputs use the official 8-5 crop, center-cropped to 3:2 and resized to 720x480 with LANCZOS.
- Contact sheet: `vbench_lotpass_pilot/selection/contact_sheet.png`.
- Metadata checksums and per-image hashes are recorded in `vbench_lotpass_pilot/selection/selection_manifest.json`.
- Selection manifest SHA-256 (approval reference): `7644474072e17a69dc4aa6e6ce8a2f5bec73aee0e5eb5f4468e8b286f43b1709`.

## Selected images

| ID | Scene | Official image | Canonical input |
|---|---|---|---|
| VB001 | architecture | a city street at night during a snow storm.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB001.png` |
| VB002 | architecture | a beach with a lot of buildings on the side of a cliff.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB002.png` |
| VB003 | architecture | a pile of wood sitting next to a log house.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB003.png` |
| VB004 | indoor | a living room with a couch, table, and a window.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB004.png` |
| VB005 | indoor | an outdoor dining area surrounded by plants and a brick walkway.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB005.png` |
| VB006 | indoor | a living room filled with furniture next to a stone wall.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB006.png` |
| VB007 | scenery | a foggy field that has trees in the grass.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB007.png` |
| VB008 | scenery | several hot air balloons flying over a city.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB008.png` |
| VB009 | scenery | a flock of birds flying over a tree at sunset.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB009.png` |
| VB010 | abstract | a view of a star trail in the night sky.jpg | `vbench_lotpass_pilot/images/canonical_clean/VB010.png` |

## Forty official prompts

| Case | Motion | Prompt (copied verbatim) |
|---|---|---|
| VB001_static | static | a city street at night during a snow storm, camera static |
| VB001_pan | pan_left | a city street at night during a snow storm, camera pans left |
| VB001_tilt | tilt_up | a city street at night during a snow storm, camera tilts up |
| VB001_zoom | zoom_in | a city street at night during a snow storm, camera zooms in |
| VB002_static | static | a beach with a lot of buildings on the side of a cliff, camera static |
| VB002_pan | pan_right | a beach with a lot of buildings on the side of a cliff, camera pans right |
| VB002_tilt | tilt_down | a beach with a lot of buildings on the side of a cliff, camera tilts down |
| VB002_zoom | zoom_out | a beach with a lot of buildings on the side of a cliff, camera zooms out |
| VB003_static | static | a pile of wood sitting next to a log house, camera static |
| VB003_pan | pan_left | a pile of wood sitting next to a log house, camera pans left |
| VB003_tilt | tilt_up | a pile of wood sitting next to a log house, camera tilts up |
| VB003_zoom | zoom_in | a pile of wood sitting next to a log house, camera zooms in |
| VB004_static | static | a living room with a couch, table, and a window, camera static |
| VB004_pan | pan_right | a living room with a couch, table, and a window, camera pans right |
| VB004_tilt | tilt_down | a living room with a couch, table, and a window, camera tilts down |
| VB004_zoom | zoom_out | a living room with a couch, table, and a window, camera zooms out |
| VB005_static | static | an outdoor dining area surrounded by plants and a brick walkway, camera static |
| VB005_pan | pan_left | an outdoor dining area surrounded by plants and a brick walkway, camera pans left |
| VB005_tilt | tilt_up | an outdoor dining area surrounded by plants and a brick walkway, camera tilts up |
| VB005_zoom | zoom_in | an outdoor dining area surrounded by plants and a brick walkway, camera zooms in |
| VB006_static | static | a living room filled with furniture next to a stone wall, camera static |
| VB006_pan | pan_right | a living room filled with furniture next to a stone wall, camera pans right |
| VB006_tilt | tilt_down | a living room filled with furniture next to a stone wall, camera tilts down |
| VB006_zoom | zoom_out | a living room filled with furniture next to a stone wall, camera zooms out |
| VB007_static | static | a foggy field that has trees in the grass, camera static |
| VB007_pan | pan_left | a foggy field that has trees in the grass, camera pans left |
| VB007_tilt | tilt_up | a foggy field that has trees in the grass, camera tilts up |
| VB007_zoom | zoom_in | a foggy field that has trees in the grass, camera zooms in |
| VB008_static | static | several hot air balloons flying over a city, camera static |
| VB008_pan | pan_right | several hot air balloons flying over a city, camera pans right |
| VB008_tilt | tilt_down | several hot air balloons flying over a city, camera tilts down |
| VB008_zoom | zoom_out | several hot air balloons flying over a city, camera zooms out |
| VB009_static | static | a flock of birds flying over a tree at sunset, camera static |
| VB009_pan | pan_left | a flock of birds flying over a tree at sunset, camera pans left |
| VB009_tilt | tilt_up | a flock of birds flying over a tree at sunset, camera tilts up |
| VB009_zoom | zoom_in | a flock of birds flying over a tree at sunset, camera zooms in |
| VB010_static | static | a view of a star trail in the night sky, camera static |
| VB010_pan | pan_right | a view of a star trail in the night sky, camera pans right |
| VB010_tilt | tilt_down | a view of a star trail in the night sky, camera tilts down |
| VB010_zoom | zoom_out | a view of a star trail in the night sky, camera zooms out |

## Files

- `official_camera_motion_all.csv`: all official camera-motion metadata records.
- `eligible_images.csv`: per-image seven-variant eligibility audit.
- `selected_10_images.csv`: selected image IDs, provenance, and paths.
- `vbench_40pairs.csv`: selected image-prompt pairs.
- `contact_sheet.png`: canonical generator-input previews.
- `selection_manifest.json`: seeds, metadata checksums, transforms, and image hashes.

**Approval gate:** Stage B must not begin until the user explicitly approves this exact selection manifest. Approval should reference the SHA-256 of `selection_manifest.json`.
