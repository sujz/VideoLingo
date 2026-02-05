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

console = Console()


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
    batch_output_dir: str = "batch/output",
    output_file: str = "batch/output/merged_batch_summary.mp4",
):
    """Merge per-video summaries under batch/output/*/batch_summary into one final video.

    - Skips batch/output/ERROR
    - Prefers merged_clips_dub.mp4 if present, else merged_clips.mp4
    - Sorts inputs by directory modification time (processing order in sequential batch)
    """
    base = Path(batch_output_dir)
    if not base.exists():
        console.print(Panel(f"No batch output directory: {batch_output_dir}", border_style="yellow"))
        return None

    candidates = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name == "ERROR" or child.name.startswith("."):
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

def process_batch():
    if not check_settings():
        raise Exception("Settings check failed")

    df = pd.read_excel('batch/tasks_setting.xlsx')
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
                status, error_step, error_message = process_video(video_file, dubbing, is_retry)
                status_msg = "Done" if status else f"Error: {error_step} - {error_message}"
            except Exception as e:
                status_msg = f"Error: Unhandled exception - {str(e)}"
                console.print(f"[bold red]Error processing {video_file}: {status_msg}")
            finally:
                update_key('whisper.language', original_source_lang)
                update_key('target_language', original_target_lang)
                
                df.at[index, 'Status'] = status_msg
                df.to_excel('batch/tasks_setting.xlsx', index=False)
                
                gc.collect()
                
                time.sleep(1)
        else:
            print(f"Skipping task: {row['Video File']} - Status: {row['Status']}")

    console.print(Panel("All tasks processed!\nCheck out in `batch/output`!", 
                       title="[bold green]Batch Processing Complete", expand=False))

    # Tail step: merge per-video summaries (skip ERROR outputs)
    try:
        merge_batch_summaries()
    except Exception:
        pass

if __name__ == "__main__":
    process_batch()