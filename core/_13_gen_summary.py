import os
import json
import subprocess
import math
from core.utils import rprint, load_key
from core.utils.models import _5_WITH_TIMESTAMPS, _OUTPUT_DIR, _4_1_TERMINOLOGY
import pandas as pd

SUMMARY_DIR = os.path.join(_OUTPUT_DIR, "summary")


def sanitize_filename(name: str) -> str:
    return ''.join(c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in name)


def score_line(text: str, terms: list) -> float:
    """Simple heuristic score: term matches (higher) + text length weight."""
    if not isinstance(text, str):
        return 0.0
    text_lower = text.lower()
    term_score = 0
    for t in terms:
        if not t:
            continue
        if isinstance(t, dict):
            check = t.get('tgt', '')
        else:
            check = str(t)
        if check and check.lower() in text_lower:
            term_score += 1
    length_score = min(len(text.split()), 40) / 40.0
    return term_score * 2.0 + length_score


def gen_video_summary(video_file: str = None):
    """Generate a summary video using budget-driven selection + greedy MMR de-dup.

    - Reads translated subtitles with numeric timestamps from `_5_WITH_TIMESTAMPS`.
    - Scores each subtitle line and selects segments under a duration budget computed from config.
    - Uses greedy MMR to prefer high-density, diverse segments.
    - Preserves original temporal order when concatenating clips.
    """
    from core._1_ytdlp import find_video_files
    from core.utils.models import _4_1_TERMINOLOGY

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # find input video
    if video_file is None:
        video_file = find_video_files()

    # load subtitles with timestamps produced by _6_gen_sub
    if not os.path.exists(_5_WITH_TIMESTAMPS):
        rprint(f"[yellow]⚠️ Subtitle file {_5_WITH_TIMESTAMPS} not found, cannot generate summary.[/yellow]")
        return None

    try:
        df = pd.read_excel(_5_WITH_TIMESTAMPS)
    except Exception as e:
        rprint(f"[red]❌ Failed to read subtitles with timestamps: {e}[/red]")
        return None

    # expected columns: Translation, start_time, end_time
    if 'start_time' not in df.columns or 'end_time' not in df.columns or 'Translation' not in df.columns:
        rprint(f"[red]❌ Subtitle file {_5_WITH_TIMESTAMPS} missing required columns ('start_time','end_time','Translation').[/red]")
        return None

    # load terms
    terms = []
    if os.path.exists(_4_1_TERMINOLOGY):
        try:
            with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as f:
                terms = json.load(f).get('terms', [])
        except Exception:
            terms = []

    # score lines using the Translation column (do not modify text)
    df['score'] = df['Translation'].apply(lambda t: score_line(t, terms))

    # ensure numeric start/end columns (they were saved in seconds by _6_gen_sub)
    df['start_s'] = df['start_time'].astype(float)
    df['end_s'] = df['end_time'].astype(float)
    df['duration'] = (df['end_s'] - df['start_s']).clip(lower=0.0)

    # Load summary config
    try:
        cfg = load_key('summary')
        ratio = float(cfg.get('ratio', 0.06))
        min_seconds = float(cfg.get('min_seconds', 8))
        max_seconds = float(cfg.get('max_seconds', 180))
        padding = float(cfg.get('padding', 0.4))
        method = cfg.get('method', 'greedy_mmr')
        mmr_lambda = float(cfg.get('mmr_lambda', 0.7))
    except Exception:
        ratio = 0.06
        min_seconds = 8.0
        max_seconds = 180.0
        padding = 0.4
        method = 'greedy_mmr'
        mmr_lambda = 0.7

    # video length estimate: max end time
    video_len = float(df['end_s'].max()) if not df['end_s'].isnull().all() else 0.0
    budget = max(min_seconds, min(max_seconds, ratio * (video_len if video_len > 0 else 1)))

    # prepare candidates: skip zero-duration lines
    candidates = df[df['duration'] > 0].copy()
    if candidates.empty:
        rprint('[yellow]⚠️ No valid subtitle candidates for summary.[/yellow]')
        return None

    # compute score density
    candidates['score_density'] = candidates['score'] / (candidates['duration'] + 1e-6)

    # normalize density for MMR relevance
    max_density = candidates['score_density'].max() if candidates['score_density'].max() > 0 else 1.0
    candidates = candidates.reset_index(drop=True)

    # helper: similarity between two texts (token overlap)
    def text_similarity(a: str, b: str) -> float:
        if not isinstance(a, str) or not isinstance(b, str):
            return 0.0
        sa = set([w for w in a.lower().split() if w])
        sb = set([w for w in b.lower().split() if w])
        if not sa or not sb:
            return 0.0
        inter = sa.intersection(sb)
        # normalized by smaller set to be conservative
        return len(inter) / float(min(len(sa), len(sb)))

    # selection via greedy MMR (maximize relevance - diversity penalty)
    selected_idxs = []
    remaining = set(candidates.index.tolist())
    total_selected_dur = 0.0

    # precompute relevance
    candidates['rel'] = candidates['score_density'] / max_density

    while remaining and total_selected_dur < budget:
        best_idx = None
        best_score = -math.inf
        for idx in list(remaining):
            dur = float(candidates.at[idx, 'duration'])
            if total_selected_dur + dur > budget:
                continue
            rel = float(candidates.at[idx, 'rel'])
            if not selected_idxs:
                mmr_score = mmr_lambda * rel
            else:
                # compute max similarity to already selected
                sims = [text_similarity(candidates.at[idx, 'Translation'], candidates.at[sid, 'Translation']) for sid in selected_idxs]
                max_sim = max(sims) if sims else 0.0
                mmr_score = mmr_lambda * rel - (1 - mmr_lambda) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected_idxs.append(best_idx)
        remaining.remove(best_idx)
        total_selected_dur += float(candidates.at[best_idx, 'duration'])

    if not selected_idxs:
        rprint('[yellow]⚠️ No segments selected under budget.[/yellow]')
        return None

    # build selected segments in original temporal order
    sel_df = candidates.loc[selected_idxs].copy()
    sel_df = sel_df.sort_values('start_s').reset_index(drop=True)

    # merge overlapping / adjacent segments (if gap <= 0.2s)
    clips = []
    for _, row in sel_df.iterrows():
        s = max(0.0, row['start_s'] - padding)
        e = row['end_s'] + padding
        if clips and s <= clips[-1][1] + 0.2:
            clips[-1][1] = max(clips[-1][1], e)
        else:
            clips.append([s, e])

    # ensure total duration within budget: compute total but DO NOT trim if
    # selected full segments exceed the budget — concatenate full segments instead
    total_clipped = sum((c[1] - c[0]) for c in clips)
    if total_clipped > budget and clips:
        rprint(f"[yellow]⚠️ Selected key segments total {total_clipped:.1f}s exceeds budget {budget:.1f}s — concatenating full segments without trimming.[/yellow]")

    if not clips:
        rprint("[yellow]⚠️ No key segments found for summary.[/yellow]")
        return None

    # detect pipeline-produced dubbed audio (prefer this over config)
    dub_audio = None
    try:
        if isinstance(cfg, dict):
            dub_audio = cfg.get('dub_audio_file')
    except Exception:
        dub_audio = None
    if not dub_audio:
        default_dub = os.path.join(_OUTPUT_DIR, 'dub.mp3')
        if os.path.exists(default_dub):
            dub_audio = default_dub

    # cut clips
    clip_files = []
    for idx, (s, e) in enumerate(clips):
        duration = e - s
        out_clip = os.path.join(SUMMARY_DIR, f"clip_{idx+1}.mp4")
        # ffmpeg: use -ss before -i for faster seek, then -t for duration
        # Re-encode each clip to ensure consistent video/audio streams
        if dub_audio and os.path.exists(dub_audio):
            # map video from source and audio from dubbed audio
            cmd = [
                'ffmpeg', '-y', '-ss', str(s), '-i', video_file,
                '-ss', str(s), '-i', dub_audio,
                '-t', str(duration), '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k', out_clip
            ]
        else:
            cmd = [
                'ffmpeg', '-y', '-ss', str(s), '-i', video_file, '-t', str(duration),
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k', out_clip
            ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clip_files.append(out_clip)
        except Exception as e:
            rprint(f"[red]Failed to cut clip {idx+1}: {e}[/red]")

    if not clip_files:
        rprint("[red]❌ No clips were successfully cut.[/red]")
        return None

    # write concat list
    concat_file = os.path.join(SUMMARY_DIR, 'concat_list.txt')
    with open(concat_file, 'w', encoding='utf-8') as f:
        for cf in clip_files:
            f.write(f"file '{os.path.abspath(cf)}'\n")

    out_summary = os.path.join(SUMMARY_DIR, sanitize_filename(os.path.splitext(os.path.basename(video_file))[0]) + '_summary.mp4')
    # Concat and re-encode final output to avoid codec/container mismatch
    cmd_concat = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', out_summary
    ]
    try:
        subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rprint(f"[green]✅ Summary video generated: {out_summary}[/green]")

        # Build subtitle file for the summary (shift original selected subtitles
        # into the new summary timeline) so we can burn Chinese translations.
        srt_file = os.path.join(SUMMARY_DIR, 'summary_subs.srt')
        try:
            # sel_df contains selected subtitle rows (with Translation,start_s,end_s)
            entries = []
            cum_offset = 0.0
            for (clip_s, clip_e), clip_file in zip(clips, clip_files):
                clip_duration = clip_e - clip_s
                # find rows overlapping this clip
                overlaps = sel_df[(sel_df['start_s'] < clip_e) & (sel_df['end_s'] > clip_s)].copy()
                if overlaps.empty:
                    cum_offset += clip_duration
                    continue
                for _, r in overlaps.iterrows():
                    new_s = max(r['start_s'], clip_s) - clip_s + cum_offset
                    new_e = min(r['end_s'], clip_e) - clip_s + cum_offset
                    entries.append((new_s, new_e, r['Translation']))
                cum_offset += clip_duration

            # write SRT
            def fmt_ts(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

            with open(srt_file, 'w', encoding='utf-8') as sf:
                for i, (ns, ne, text) in enumerate(entries, start=1):
                    sf.write(f"{i}\n")
                    sf.write(f"{fmt_ts(ns)} --> {fmt_ts(ne)}\n")
                    sf.write(str(text).replace('\n', ' ') + "\n\n")

            # burn subtitles into final summary (produce a _sub file)
            out_summary_sub = os.path.splitext(out_summary)[0] + '_sub.mp4'
            cmd_burn = [
                'ffmpeg', '-y', '-i', out_summary, '-vf', f"subtitles={os.path.abspath(srt_file)}",
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-c:a', 'copy', out_summary_sub
            ]
            subprocess.run(cmd_burn, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rprint(f"[green]✅ Summary with burned subtitles: {out_summary_sub}[/green]")
            return {'summary_file': out_summary_sub, 'srt': srt_file}
        except Exception as e:
            rprint(f"[yellow]⚠️ Subtitle burn failed or no subtitles available: {e}[/yellow]")
            return {'summary_file': out_summary}
    except Exception as e:
        rprint(f"[red]❌ Failed to concat clips: {e}[/red]")
        return None


if __name__ == '__main__':
    gen_video_summary()
