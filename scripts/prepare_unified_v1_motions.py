"""Convert the user's annotated CSV intervals into the v1 AMP Recovery clips.

Run inside the project's ``amp`` conda environment:
  python scripts/prepare_unified_v1_motions.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_DIR = ROOT / "src/assets/motions/g1/amp_unified_v1/Recovery"
CSV1 = Path("/home/xyang/Downloads/fallAndGetUp1_subject1.csv")
CSV2 = Path("/home/xyang/Downloads/fallAndGetUp2_subject2.csv")

# Inclusive CSV line ranges exactly as annotated by the user.
SEGMENTS = [
    (CSV1, 491, 575, "subject1_01_supine_side_getup"),
    (CSV1, 698, 806, "subject1_02_prone_side_getup"),
    (CSV1, 2080, 2212, "subject1_03_supine_side_getup"),
    (CSV1, 2630, 2745, "subject1_04_supine_left_getup_walk"),
    (CSV1, 2905, 3038, "subject1_05_supine_right_getup_walk"),
    (CSV1, 3230, 3350, "subject1_06_supine_straight_getup_walk"),
    (CSV1, 4290, 4400, "subject1_07_supine_straight_getup_run"),
    (CSV1, 4553, 4660, "subject1_08_prone_straight_getup_run"),
    (CSV2, 350, 580, "subject2_01_supine_straight_getup_high_quality"),
    (CSV2, 800, 1042, "subject2_02_prone_straight_getup_high_quality"),
    (CSV2, 1200, 1380, "subject2_03_supine_side_getup_high_quality"),
]


def main() -> None:
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    converter = ROOT / "scripts/csv_to_npz.py"
    manifest = []
    for source, start, end, name in SEGMENTS:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = RECOVERY_DIR / f"{name}.npz"
        command = [
            sys.executable, str(converter),
            "--input-file", str(source),
            "--output-name", target.name,
            "--output-dir", str(RECOVERY_DIR),
            "--input-fps", "30",
            "--output-fps", "50",
            "--device", "cpu",
            "--line-range", f"({start},{end})",
        ]
        print("\n$", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        manifest.append({
            "source_csv": str(source),
            "start_line_inclusive": start,
            "end_line_inclusive": end,
            "label": name,
            "output_npz": str(target.relative_to(ROOT)),
            "input_fps": 30,
            "output_fps": 50,
        })
    manifest_path = RECOVERY_DIR.parent / "manifest.json"
    manifest_path.write_text(json.dumps({
        "dataset": "amp_unified_v1",
        "user_annotated_ranges_used_as_is": True,
        "segments": manifest,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
