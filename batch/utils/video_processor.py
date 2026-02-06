import os
from core.st_utils.imports_and_utils import *
from core.utils.onekeycleanup import cleanup
from core.utils.onekeycleanup import sanitize_filename
from core.utils import load_key
import shutil
from functools import partial
from rich.panel import Panel
from rich.console import Console
from core import *
from core._13_identify_key_segments import identify_key_segments_from_subtitles
from core._14_extract_video_clips import (
    extract_video_clips_by_segments,
    merge_all_video_clips,
    create_dubbed_summary_from_segments,
)

console = Console()

INPUT_DIR = 'batch/input'
OUTPUT_DIR = 'output'
SAVE_DIR = 'batch/output'
ERROR_OUTPUT_DIR = 'batch/output/ERROR'
YTB_RESOLUTION_KEY = "ytb_resolution"

def process_video(file, dubbing=False, is_retry=False, step_cb=None):
    # Prepare workspace folder for this task.
    # - Normal run: wipe and recreate output/ for a clean pipeline.
    # - Retry run: keep existing files if present, but still ensure output/ exists.
    if step_cb:
        try:
            step_cb("🧹 Preparing output folder")
        except Exception:
            pass

    if is_retry:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    else:
        prepare_output_folder(OUTPUT_DIR)
    
    text_steps = [
        ("🎥 Processing input file", partial(process_input_file, file)),
        ("🎙️ Transcribing with Whisper", partial(_2_asr.transcribe)),
        ("✂️ Splitting sentences", split_sentences),
        ("📝 Summarizing and translating", summarize_and_translate),
        ("⚡ Processing and aligning subtitles", process_and_align_subtitles),
        ("🎬 Merging subtitles to video", _7_sub_into_vid.merge_subtitles_to_video),
    ]
    
    if dubbing:
        dubbing_steps = [
            ("🔊 Generating audio tasks", gen_audio_tasks),
            ("🎵 Extracting reference audio", _9_refer_audio.extract_refer_audio_main),
            ("🗣️ Generating audio", _10_gen_audio.gen_audio),
            ("🔄 Merging full audio", _11_merge_audio.merge_full_audio),
            ("🎞️ Merging dubbing to video", _12_dub_to_vid.merge_video_audio),
        ]
        text_steps.extend(dubbing_steps)
    
    current_step = ""
    for step_name, step_func in text_steps:
        current_step = step_name
        if step_cb:
            try:
                step_cb(step_name)
            except Exception:
                pass
        for attempt in range(3):
            try:
                console.print(Panel(
                    f"[bold green]{step_name}[/]",
                    subtitle=f"Attempt {attempt + 1}/3" if attempt > 0 else None,
                    border_style="blue"
                ))
                result = step_func()
                if result is not None:
                    globals().update(result)
                break
            except Exception as e:
                if attempt == 2:
                    error_panel = Panel(
                        f"[bold red]Error in step '{current_step}':[/]\n{str(e)}",
                        border_style="red"
                    )
                    console.print(error_panel)
                    cleanup(ERROR_OUTPUT_DIR)
                    return False, current_step, str(e), None
                console.print(Panel(
                    f"[yellow]Attempt {attempt + 1} failed. Retrying...[/]",
                    border_style="yellow"
                ))
    
    console.print(Panel("[bold green]All steps completed successfully! 🎉[/]", border_style="green"))
    # After full pipeline (including dubbing if requested), generate summary
    # so it can use the merged dubbed audio and burn Chinese subtitles.
    try:
        for attempt in range(2):
            try:
                console.print(Panel(f"[bold green]📝 Generating video summary (post-processing)[/]", border_style="blue"))
                # Use identify + clip extraction flow to create summary
                key_segments = identify_key_segments_from_subtitles()
                res = None
                if key_segments:
                    clips = extract_video_clips_by_segments(key_segments)
                    merged = None
                    try:
                        merged = merge_all_video_clips()
                    except Exception:
                        pass
                    try:
                        create_dubbed_summary_from_segments(key_segments, merged_video=merged)
                    except Exception:
                        pass
                    res = {'clips': clips if 'clips' in locals() else None, 'merged': merged}
                if res is not None:
                    globals().update(res)
                break
            except Exception:
                if attempt == 1:
                    console.print(Panel(f"[bold yellow]Warning: summary generation failed after dubbing.[/]", border_style="yellow"))
    except Exception:
        pass

    # Predict the output directory name deterministically (same as cleanup()).
    task_output_dir = None
    try:
        video_file = globals().get('video_file')
        if isinstance(video_file, str) and video_file:
            base = os.path.splitext(os.path.basename(video_file))[0]
            task_output_dir = os.path.join(SAVE_DIR, sanitize_filename(base))
    except Exception:
        task_output_dir = None

    cleanup(SAVE_DIR)
    return True, "", "", task_output_dir

def prepare_output_folder(output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

def process_input_file(file):
    if file.startswith('http'):
        _1_ytdlp.download_video_ytdlp(file, resolution=load_key(YTB_RESOLUTION_KEY))
        video_file = _1_ytdlp.find_video_files()
        try:
            # Marker used to map URL -> batch/output/<dir>
            with open(os.path.join(OUTPUT_DIR, 'source_url.txt'), 'w', encoding='utf-8') as f:
                f.write(str(file).strip() + "\n")
        except Exception:
            pass
    else:
        input_file = os.path.join('batch', 'input', file)
        output_file = os.path.join(OUTPUT_DIR, file)
        shutil.copy(input_file, output_file)
        video_file = output_file
        try:
            with open(os.path.join(OUTPUT_DIR, 'source_url.txt'), 'w', encoding='utf-8') as f:
                f.write(str(file).strip() + "\n")
        except Exception:
            pass
    return {'video_file': video_file}

def split_sentences():
    _3_1_split_nlp.split_by_spacy()
    _3_2_split_meaning.split_sentences_by_meaning()

def summarize_and_translate():
    _4_1_summarize.get_summary()
    _4_2_translate.translate_all()

def process_and_align_subtitles():
    _5_split_sub.split_for_sub_main()
    _6_gen_sub.align_timestamp_main()

def gen_audio_tasks():
    _8_1_audio_task.gen_audio_task_main()
    _8_2_dub_chunks.gen_dub_chunks()
