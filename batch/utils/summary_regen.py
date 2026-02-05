from __future__ import annotations

import os
from pathlib import Path
from shutil import copy2, copytree, rmtree


def _pick_main_video_file(task_dir: Path) -> Path | None:
    exts = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}
    if not task_dir.exists():
        return None

    excluded = {
        "output_sub.mp4",
        "output_dub.mp4",
        "batch_summary_all.mp4",
        "merged_batch_summary.mp4",
    }

    candidates: list[Path] = []
    for p in task_dir.iterdir():
        if not p.is_file():
            continue
        if p.name in excluded:
            continue
        if p.suffix.lower() not in exts:
            continue
        if p.name.startswith("merged_clips"):
            continue
        candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)
    return candidates[0]


def regenerate_summary_from_edited_subtitle(task_output_dir: str) -> dict:
    """Regenerate subtitle-burned video + per-video summary using edited trans.srt.

    This is intentionally implemented as an independent utility module so it doesn't
    require changing the normal pipeline.

    Inputs (from task_output_dir):
    - trans.srt (edited Chinese subtitles)
    - src.srt
    - original downloaded video file

    Outputs (written/overwritten under task_output_dir):
    - output_sub.mp4
    - batch_summary/* (merged_clips*.mp4 + metadata)
    """

    task_dir = Path(task_output_dir)
    if not task_dir.exists():
        raise FileNotFoundError(f"Task output dir not found: {task_output_dir}")

    trans_srt = task_dir / "trans.srt"
    src_srt = task_dir / "src.srt"
    if not trans_srt.exists():
        raise FileNotFoundError(f"Missing edited subtitle: {trans_srt}")
    if not src_srt.exists():
        raise FileNotFoundError(f"Missing source subtitle: {src_srt}")

    main_video = _pick_main_video_file(task_dir)
    if not main_video:
        raise FileNotFoundError(f"No source video found under: {task_output_dir}")

    # Prepare working output/ folder (core code expects this location).
    if os.path.exists("output"):
        rmtree("output", ignore_errors=True)
    os.makedirs("output", exist_ok=True)

    copy2(str(main_video), os.path.join("output", main_video.name))
    copy2(str(src_srt), os.path.join("output", "src.srt"))
    copy2(str(trans_srt), os.path.join("output", "trans.srt"))

    # Run minimal post-translate pipeline using edited subtitles.
    from core import _7_sub_into_vid
    from core._13_identify_key_segments import identify_key_segments_from_subtitles
    from core._14_extract_video_clips import (
        extract_video_clips_by_segments,
        merge_all_video_clips,
        create_dubbed_summary_from_segments,
    )

    _7_sub_into_vid.merge_subtitles_to_video()

    key_segments = identify_key_segments_from_subtitles()
    extract_video_clips_by_segments(key_segments)
    merged = merge_all_video_clips()
    try:
        create_dubbed_summary_from_segments(key_segments, merged_video=merged)
    except Exception:
        pass

    out_sub_video = Path("output") / "output_sub.mp4"
    if out_sub_video.exists():
        copy2(str(out_sub_video), str(task_dir / "output_sub.mp4"))

    out_summary_dir = Path("output") / "batch_summary"
    if out_summary_dir.exists():
        dst_summary_dir = task_dir / "batch_summary"
        if dst_summary_dir.exists():
            rmtree(dst_summary_dir, ignore_errors=True)
        copytree(out_summary_dir, dst_summary_dir)

    return {
        "task_dir": str(task_dir),
        "output_sub": str(task_dir / "output_sub.mp4"),
        "summary_video": str(task_dir / "batch_summary" / "merged_clips.mp4"),
    }
