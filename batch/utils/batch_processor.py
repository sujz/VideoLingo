import os
import gc
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from batch.utils.settings_check import check_settings
from core.utils.config_utils import load_key, update_key
import pandas as pd
from rich.console import Console
from rich.panel import Panel
import time
import shutil

console = Console()

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
    # collect tasks to run
    tasks = []
    total_tasks = len(df)
    for index, row in df.iterrows():
        if pd.isna(row['Status']) or 'Error' in str(row['Status']):
            video_file = row['Video File']
            is_retry = not pd.isna(row['Status']) and 'Error' in str(row['Status'])
            tasks.append((index, video_file, row))
        else:
            print(f"Skipping task: {row['Video File']} - Status: {row['Status']}")

    if not tasks:
        console.print(Panel("No tasks to process.", title="[bold yellow]Batch", expand=False))
        return

    # prepare worker commands (restore ERROR folders if retry)
    cmds = []
    for index, video_file, row in tasks:
        # restore from ERROR if needed
        if not pd.isna(row['Status']) and 'Error' in str(row['Status']):
            console.print(Panel(f"Retrying failed task: {video_file}\nTask {index + 1}/{total_tasks}", title="[bold yellow]Retry Task", expand=False))
            error_folder = os.path.join('batch', 'output', 'ERROR', os.path.splitext(str(video_file))[0])
            if os.path.exists(error_folder):
                os.makedirs('output', exist_ok=True)
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
            console.print(Panel(f"Now processing task: {video_file}", title="[bold blue]Current Task", expand=False))

        source_language = row['Source Language']
        target_language = row['Target Language']
        dubbing = 0 if pd.isna(row['Dubbing']) else int(row['Dubbing'])
        is_retry_flag = 1 if (not pd.isna(row['Status']) and 'Error' in str(row['Status'])) else 0

        # prepare subprocess worker command
        cmd = [sys.executable, '-m', 'batch.utils.video_worker', '--video', str(video_file), '--dubbing', str(dubbing), '--is_retry', str(is_retry_flag), '--source', str(source_language) if not pd.isna(source_language) else '', '--target', str(target_language) if not pd.isna(target_language) else '', '--index', str(index)]
        cmds.append((index, video_file, cmd))

    max_workers = min(len(cmds), (os.cpu_count() or 2))
    console.print(Panel(f"Starting parallel processing with {max_workers} workers...", title="[bold green]Batch Parallel", expand=False))

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        future_map = {exc.submit(subprocess.run, cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT): (idx, vf, cmd) for idx, vf, cmd in cmds}
        for fut in as_completed(future_map):
            idx, vf, cmd = future_map[fut]
            try:
                proc = fut.result()
                out = proc.stdout.decode('utf-8') if proc.stdout else ''
                if proc.returncode == 0:
                    status_msg = 'Done'
                else:
                    status_msg = f"Error: Worker exit {proc.returncode}. See worker output."
                # update dataframe
                df.at[idx, 'Status'] = status_msg
                df.to_excel('batch/tasks_setting.xlsx', index=False)
                console.print(f"Task {vf}: {status_msg}")
            except Exception as e:
                df.at[idx, 'Status'] = f"Error: Exception in worker - {e}"
                df.to_excel('batch/tasks_setting.xlsx', index=False)
                console.print(f"Task {vf} failed with exception: {e}")
            finally:
                gc.collect()
                time.sleep(1)

    console.print(Panel("All tasks processed!\nCheck out in `batch/output`!", title="[bold green]Batch Processing Complete", expand=False))

if __name__ == "__main__":
    process_batch()