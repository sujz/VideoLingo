import os
import gc
import glob
import subprocess
from pathlib import Path
from batch.utils.settings_check import check_settings
from batch.utils.video_processor import process_video
from core.utils.config_utils import load_key, update_key
import pandas as pd
from rich.console import Console
from rich.panel import Panel
import time
import shutil
import threading
import zipfile

from batch.utils.summary_regen import regenerate_summary_from_edited_subtitle

console = Console()


_TASKS_IO_LOCK = threading.Lock()


TASKS_XLSX_PATH = "batch/tasks_setting.xlsx"
SAVE_DIR = "batch/output"
ERROR_OUTPUT_DIR = "batch/output/ERROR"
OUTPUT_DIR_COL = "Output Dir"


def _probe_video_wh(path: str):
    """Return (width, height) for a video using ffprobe, or (None, None) on failure."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        if "x" in out:
            w, h = out.split("x", 1)
            return int(w), int(h)
    except Exception:
        pass
    return None, None


def merge_batch_summaries(
    batch_output_dir: str = SAVE_DIR,
    error_output_dir: str = ERROR_OUTPUT_DIR,
    output_file: str = "batch/output/batch_summary_all.mp4",
    inputs: list[str] | None = None,
):
    """Merge per-video summaries under batch/output/*/batch_summary into one final video.

    - Skips batch/output/ERROR
    - Prefers merged_clips_dub.mp4 if present, else merged_clips.mp4
    - Sorts inputs by directory modification time (processing order in sequential batch)
    """
    if inputs is None:
        base = Path(batch_output_dir)
        if not base.exists():
            console.print(Panel(f"No batch output directory: {batch_output_dir}", border_style="yellow"))
            return None

        candidates = []
        error_dir = Path(error_output_dir)

        for child in base.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith("."):
                continue

            if error_dir.exists():
                try:
                    if child.resolve() == error_dir.resolve():
                        continue
                except Exception:
                    pass

            # Back-compat: also skip by name.
            if child.name == "ERROR":
                continue

            summary_dir = child / "batch_summary"
            if not summary_dir.exists():
                continue

            preferred = summary_dir / "merged_clips_dub.mp4"
            fallback = summary_dir / "merged_clips.mp4"
            if preferred.exists():
                summary_path = preferred
            elif fallback.exists():
                summary_path = fallback
            else:
                matches = sorted(summary_dir.glob("merged_clips*.mp4"))
                if not matches:
                    continue
                summary_path = matches[0]

            try:
                dir_mtime = child.stat().st_mtime
            except Exception:
                dir_mtime = 0
            candidates.append((dir_mtime, str(summary_path)))

        if not candidates:
            console.print(Panel("No per-video summaries found to merge.", border_style="yellow"))
            return None

        candidates.sort(key=lambda x: x[0])
        inputs = [p for _, p in candidates]
    else:
        inputs = [str(p).strip() for p in inputs if str(p).strip()]
        inputs = [p for p in inputs if os.path.exists(p)]
        if not inputs:
            console.print(Panel("No input summaries provided to merge.", border_style="yellow"))
            return None

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write an input manifest for debugging
    manifest = os.path.splitext(output_file)[0] + "_inputs.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        for p in inputs:
            f.write(p + "\n")

    # Build robust concat (re-encode) so differing resolutions/codecs won't break merging.
    target_w, target_h = _probe_video_wh(inputs[0])
    if not target_w or not target_h:
        target_w, target_h = 1280, 720

    filter_parts = []
    for i in range(len(inputs)):
        filter_parts.append(
            f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}]"
        )
        filter_parts.append(
            f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"aresample=async=1[a{i}]"
        )
    concat_in = "".join([f"[v{i}][a{i}]" for i in range(len(inputs))])
    filter_parts.append(f"{concat_in}concat=n={len(inputs)}:v=1:a=1[v][a]")
    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd.extend(["-i", p])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            output_file,
        ]
    )

    console.print(
        Panel(
            f"Merging {len(inputs)} video summaries into:\n{output_file}\n\nManifest:\n{manifest}",
            title="[bold green]Batch Summary Merge",
            expand=False,
        )
    )

    try:
        subprocess.run(cmd, check=True)
        return output_file
    except Exception as e:
        console.print(Panel(f"Warning: failed to merge batch summaries: {e}", border_style="yellow"))
        return None


def record_and_update_config(source_language, target_language):
    original_source_lang = load_key('whisper.language')
    original_target_lang = load_key('target_language')
    
    if source_language and not pd.isna(source_language):
        update_key('whisper.language', source_language)
    if target_language and not pd.isna(target_language):
        update_key('target_language', target_language)
    
    return original_source_lang, original_target_lang


def _safe_str(val) -> str:
    return "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)


def load_tasks_df(path: str = TASKS_XLSX_PATH) -> pd.DataFrame:
    """Load tasks_setting.xlsx, creating an empty template if missing."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(columns=['Video File', 'Source Language', 'Target Language', 'Dubbing', 'Status', OUTPUT_DIR_COL])
        df.to_excel(path, index=False)
        return df

    bak_path = path + ".bak"
    last_err: Exception | None = None
    with _TASKS_IO_LOCK:
        for attempt in range(6):
            try:
                df = pd.read_excel(path)
                break
            except zipfile.BadZipFile as e:
                # Likely read during a concurrent write; retry.
                last_err = e
                time.sleep(0.15 * (attempt + 1))
            except Exception as e:
                last_err = e
                time.sleep(0.10 * (attempt + 1))
        else:
            # Fallback to backup if present.
            if os.path.exists(bak_path):
                try:
                    df = pd.read_excel(bak_path)
                except Exception:
                    raise last_err  # type: ignore[misc]
            else:
                raise last_err  # type: ignore[misc]

    # Back-compat: add new columns if missing.
    if OUTPUT_DIR_COL not in df.columns:
        df[OUTPUT_DIR_COL] = None
        save_tasks_df(df, path)
    return df


def save_tasks_df(df: pd.DataFrame, path: str = TASKS_XLSX_PATH) -> None:
    # Must keep a .xlsx suffix so pandas can select an Excel writer engine.
    tmp_path = path + ".tmp.xlsx"
    bak_path = path + ".bak"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with _TASKS_IO_LOCK:
        # Keep a best-effort backup of the last good file.
        if os.path.exists(path):
            try:
                shutil.copy2(path, bak_path)
            except Exception:
                pass

        # Write to a temp file then atomically replace.
        df.to_excel(tmp_path, index=False)
        os.replace(tmp_path, path)


def append_tasks(video_urls: list[str], path: str = TASKS_XLSX_PATH) -> pd.DataFrame:
    """Append URLs as tasks. Leaves language blank so config defaults apply."""
    df = load_tasks_df(path)
    existing = set(_safe_str(x) for x in df.get('Video File', []).tolist())
    new_rows = []
    for url in video_urls:
        url = _safe_str(url).strip()
        if not url or url in existing:
            continue
        new_rows.append({
            'Video File': url,
            'Source Language': '',
            'Target Language': '',
            'Dubbing': 1,
            'Status': None,
            OUTPUT_DIR_COL: None,
        })
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_tasks_df(df, path)
    return df


def _find_latest_task_output_dir(
    batch_output_dir: str = SAVE_DIR,
    error_output_dir: str = ERROR_OUTPUT_DIR,
) -> str | None:
    """Best-effort: pick the most recently modified task directory under batch/output.

    This is used to map a processed URL row to its output folder name.
    Assumes sequential processing.
    """
    base = Path(batch_output_dir)
    if not base.exists():
        return None

    error_dir = Path(error_output_dir)
    best_mtime = -1
    best_path: Path | None = None

    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith('.'):
            continue
        if child.name == 'ERROR':
            continue
        if error_dir.exists():
            try:
                if child.resolve() == error_dir.resolve():
                    continue
            except Exception:
                pass
        try:
            mtime = child.stat().st_mtime
        except Exception:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best_path = child

    return str(best_path) if best_path else None


"""NOTE:
Summary regeneration is implemented in batch/utils/summary_regen.py to keep it
independent from the normal pipeline and reduce risk.
"""


def _find_summary_video_in_task_dir(task_dir: Path) -> Path | None:
    """Prefer dubbed summary if present, else non-dubbed merged clips."""
    batch_summary = task_dir / 'batch_summary'
    if not batch_summary.exists():
        return None
    dubbed = batch_summary / 'merged_clips_dub.mp4'
    if dubbed.exists():
        return dubbed
    merged = batch_summary / 'merged_clips.mp4'
    if merged.exists():
        return merged
    return None


def merge_successful_summaries(
    batch_output_dir: str = SAVE_DIR,
    error_output_dir: str = ERROR_OUTPUT_DIR,
    output_file: str = 'batch/output/batch_summary_all.mp4',
) -> str | None:
    """Merge per-video summaries into one final summary, skipping ERROR tasks."""
    return merge_batch_summaries(
        batch_output_dir=batch_output_dir,
        error_output_dir=error_output_dir,
        output_file=output_file,
    )


def process_selected_videos(
    video_urls: list[str],
    tasks_path: str = TASKS_XLSX_PATH,
    progress_cb=None,
    merge_summaries: bool = False,
) -> None:
    """Process a specific set of URLs sequentially (batch logic) and update Status in tasks xlsx.

    progress_cb signature: (video_url: str, status: str) -> None
    """
    df = load_tasks_df(tasks_path)
    url_set = set(_safe_str(u).strip() for u in video_urls if _safe_str(u).strip())
    if not url_set:
        return

    for index, row in df.iterrows():
        video_file = _safe_str(row.get('Video File')).strip()
        if video_file not in url_set:
            continue

        # mark running
        df.at[index, 'Status'] = 'Running'
        save_tasks_df(df, tasks_path)
        if progress_cb:
            progress_cb(video_file, 'Running')

        # Make it obvious that processing actually started, even before the first heavy step.
        try:
            os.makedirs('output', exist_ok=True)
            df.at[index, 'Status'] = 'Running: 🧹 Preparing output folder'
            save_tasks_df(df, tasks_path)
            if progress_cb:
                progress_cb(video_file, 'Running: 🧹 Preparing output folder')
        except Exception:
            pass

        def _step_cb(step_name: str, _index=index, _video_file=video_file):
            try:
                current_df = load_tasks_df(tasks_path)
                current_df.at[_index, 'Status'] = f"Running: {step_name}"
                save_tasks_df(current_df, tasks_path)
                if progress_cb:
                    progress_cb(_video_file, f"Running: {step_name}")
            except Exception:
                pass

        source_language = row.get('Source Language')
        target_language = row.get('Target Language')
        original_source_lang, original_target_lang = record_and_update_config(source_language, target_language)

        try:
            dubbing = 0 if pd.isna(row.get('Dubbing')) else int(row.get('Dubbing'))
            is_retry = not pd.isna(row.get('Status')) and 'Error' in str(row.get('Status'))
            result = process_video(video_file, dubbing=bool(dubbing), is_retry=bool(is_retry), step_cb=_step_cb)
            if isinstance(result, tuple) and len(result) == 4:
                status, error_step, error_message, task_output_dir = result
            else:
                status, error_step, error_message = result
                task_output_dir = None
            status_msg = 'Done' if status else f"Error: {error_step} - {error_message}"
        except BaseException as e:
            # If the app is stopped/reloaded, avoid leaving the task stuck at "Running".
            status_msg = f"Error: Aborted - {type(e).__name__}"
            df.at[index, 'Status'] = status_msg
            try:
                save_tasks_df(df, tasks_path)
                if progress_cb:
                    progress_cb(video_file, status_msg)
            except Exception:
                pass
            raise
        except Exception as e:
            status_msg = f"Error: Unhandled exception - {str(e)}"
            console.print(f"[bold red]Error processing {video_file}: {status_msg}")
        finally:
            update_key('whisper.language', original_source_lang)
            update_key('target_language', original_target_lang)

            if status_msg == 'Done':
                if task_output_dir:
                    df.at[index, OUTPUT_DIR_COL] = task_output_dir
                else:
                    latest_dir = _find_latest_task_output_dir()
                    if latest_dir:
                        df.at[index, OUTPUT_DIR_COL] = latest_dir

            df.at[index, 'Status'] = status_msg
            save_tasks_df(df, tasks_path)
            if progress_cb:
                progress_cb(video_file, status_msg)

            gc.collect()
            time.sleep(1)

    # Post-batch: optionally merge successful summaries (skip ERROR)
    if merge_summaries:
        try:
            merged = merge_successful_summaries()
            if merged and progress_cb:
                progress_cb('SUMMARY_MERGE', f"Merged summaries -> {merged}")
        except Exception as e:
            if progress_cb:
                progress_cb('SUMMARY_MERGE', f"Summary merge failed: {e}")

def process_batch(merge_summaries: bool = False):
    if not check_settings():
        raise Exception("Settings check failed")

    df = load_tasks_df(TASKS_XLSX_PATH)
    for index, row in df.iterrows():
        if pd.isna(row['Status']) or 'Error' in str(row['Status']):
            total_tasks = len(df)
            video_file = row['Video File']
            
            if not pd.isna(row['Status']) and 'Error' in str(row['Status']):
                console.print(Panel(f"Retrying failed task: {video_file}\nTask {index + 1}/{total_tasks}", 
                                 title="[bold yellow]Retry Task", expand=False))
                
                # Restore files from batch/output/ERROR to output
                error_folder = os.path.join('batch', 'output', 'ERROR', os.path.splitext(video_file)[0])
                
                if os.path.exists(error_folder):
                    # Ensure the output folder exists
                    os.makedirs('output', exist_ok=True)
                    
                    # Copy all contents from ERROR folder for the specific video to output
                    for item in os.listdir(error_folder):
                        src_path = os.path.join(error_folder, item)
                        dst_path = os.path.join('output', item)
                        
                        if os.path.isdir(src_path):
                            if os.path.exists(dst_path):
                                shutil.rmtree(dst_path)
                            shutil.copytree(src_path, dst_path)
                        else:
                            if os.path.exists(dst_path):
                                os.remove(dst_path)
                            shutil.copy2(src_path, dst_path)
                            
                    console.print(f"[green]Restored files from ERROR folder for {video_file}")
                else:
                    console.print(f"[yellow]Warning: Error folder not found: {error_folder}")
            else:
                console.print(Panel(f"Now processing task: {video_file}\nTask {index + 1}/{total_tasks}", 
                                 title="[bold blue]Current Task", expand=False))
            
            source_language = row['Source Language']
            target_language = row['Target Language']
            
            original_source_lang, original_target_lang = record_and_update_config(source_language, target_language)
            
            try:
                dubbing = 0 if pd.isna(row['Dubbing']) else int(row['Dubbing'])
                is_retry = not pd.isna(row['Status']) and 'Error' in str(row['Status'])
                def _step_cb(step_name: str, _index=index):
                    try:
                        current_df = pd.read_excel(TASKS_XLSX_PATH)
                        current_df.at[_index, 'Status'] = f"Running: {step_name}"
                        current_df.to_excel(TASKS_XLSX_PATH, index=False)
                    except Exception:
                        pass

                result = process_video(video_file, dubbing=bool(dubbing), is_retry=bool(is_retry), step_cb=_step_cb)
                if isinstance(result, tuple) and len(result) == 4:
                    status, error_step, error_message, task_output_dir = result
                else:
                    status, error_step, error_message = result
                    task_output_dir = None
                status_msg = "Done" if status else f"Error: {error_step} - {error_message}"
            except BaseException as e:
                status_msg = f"Error: Aborted - {type(e).__name__}"
                df.at[index, 'Status'] = status_msg
                save_tasks_df(df, TASKS_XLSX_PATH)
                raise
            except Exception as e:
                status_msg = f"Error: Unhandled exception - {str(e)}"
                console.print(f"[bold red]Error processing {video_file}: {status_msg}")
            finally:
                update_key('whisper.language', original_source_lang)
                update_key('target_language', original_target_lang)

                if status_msg == 'Done':
                    if task_output_dir:
                        df.at[index, OUTPUT_DIR_COL] = task_output_dir
                    else:
                        latest_dir = _find_latest_task_output_dir()
                        if latest_dir:
                            df.at[index, OUTPUT_DIR_COL] = latest_dir
                
                df.at[index, 'Status'] = status_msg
                df.to_excel(TASKS_XLSX_PATH, index=False)
                
                gc.collect()
                
                time.sleep(1)
        else:
            print(f"Skipping task: {row['Video File']} - Status: {row['Status']}")

    # Post-batch: optionally merge successful summaries (skip ERROR)
    if merge_summaries:
        try:
            merged = merge_successful_summaries()
            if merged:
                console.print(Panel(f"Merged summaries -> {merged}", title="[bold green]Batch Summary Merge", expand=False))
        except Exception as e:
            console.print(Panel(f"Summary merge failed: {e}", title="[bold yellow]Batch Summary Merge", expand=False))

    console.print(Panel("All tasks processed!\nCheck out in `batch/output`!", 
                       title="[bold green]Batch Processing Complete", expand=False))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VideoLingo batch processor")
    parser.add_argument(
        "--merge-summaries",
        action="store_true",
        help="Merge per-video summary videos into one final video (default: off)",
    )
    args = parser.parse_args()

    process_batch(merge_summaries=bool(args.merge_summaries))