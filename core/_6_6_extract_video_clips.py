import subprocess
import os
import json
from typing import List, Dict
from pathlib import Path
from core.utils import *
from core.utils.models import *
from core._1_ytdlp import find_video_files
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def extract_video_clips_by_segments(
    key_segments: List[Dict],
    output_dir: str = "output/batch_summary/clips",
    use_copy_codec: bool = True
) -> List[str]:
    """
    使用 FFmpeg 切割关键视频片段
    
    执行点: _6_5_identify_key_segments.identify_key_segments_from_subtitles() ✓ → 本函数 ⭐
    
    Args:
        key_segments: 关键段列表，来自 identify_key_segments_from_subtitles()
                     每个段包含: start_seconds, end_seconds, duration, zh_summary 等
        output_dir: 输出目录
        use_copy_codec: 是否使用 -c copy（快速复制，无转码）
    
    Returns:
        clips_list: 生成的视频片段路径列表
    
    优点:
        1. 基于已验证的关键段时间戳
        2. 时间戳精确，与中文字幕完全对应
        3. 使用 -c copy 快速切割（无转码，速度快）
        4. 自动生成 clips_info.json 记录元数据
    """
    
    console.print(Panel(f"[bold cyan]🎬 Extracting {len(key_segments)} video clips[/bold cyan]"))
    
    # 1. 找到原始视频文件
    try:
        video_file = find_video_files()
        console.print(f"[cyan]Source video: {video_file}[/cyan]")
    except Exception as e:
        console.print(f"[red]❌ Error finding video file: {e}[/red]")
        raise
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 逐个切割片段
    clips_list = []
    clips_metadata = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task(
            "[cyan]Extracting clips...",
            total=len(key_segments)
        )
        
        for segment in key_segments:
            segment_id = segment.get('segment_id', 0)
            start_seconds = segment['start_seconds']
            end_seconds = segment['end_seconds']
            duration = segment['duration']
            
            # 输出文件名
            clip_filename = f"clip_{segment_id:02d}.mp4"
            output_file = os.path.join(output_dir, clip_filename)
            
            try:
                # FFmpeg 命令：使用时间戳直接切割，无需转码
                if use_copy_codec:
                    # 快速模式：直接复制编码
                    cmd = [
                        'ffmpeg',
                        '-i', video_file,
                        '-ss', str(start_seconds),           # 开始时间
                        '-to', str(end_seconds),             # 结束时间
                        '-c:v', 'copy',                      # 视频流：直接复制
                        '-c:a', 'copy',                      # 音频流：直接复制
                        '-y',                                # 覆盖输出文件
                        output_file
                    ]
                else:
                    # 完整转码模式（如果需要调整质量）
                    cmd = [
                        'ffmpeg',
                        '-i', video_file,
                        '-ss', str(start_seconds),
                        '-to', str(end_seconds),
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-crf', '20',
                        '-c:a', 'aac',
                        '-b:a', '128k',
                        '-y',
                        output_file
                    ]
                
                # 执行 FFmpeg
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 300秒超时
                )
                
                # 验证输出文件
                if os.path.exists(output_file):
                    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                    clips_list.append(output_file)
                    
                    # 记录元数据
                    clips_metadata.append({
                        'clip_id': segment_id,
                        'filename': clip_filename,
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'duration': duration,
                        'importance': segment['importance'],
                        'reason': segment['reason'],
                        'zh_summary': segment.get('zh_summary', ''),
                        'file_size_mb': file_size_mb,
                        'file_path': output_file
                    })
                    
                    console.print(
                        f"[green]✓[/green] Clip {segment_id}: "
                        f"{clip_filename} ({file_size_mb:.1f} MB, {duration:.1f}s)"
                    )
                else:
                    console.print(f"[red]❌ Output file not created: {output_file}[/red]")
                    raise FileNotFoundError(f"FFmpeg did not create output: {output_file}")
            
            except subprocess.TimeoutExpired:
                console.print(f"[red]❌ Timeout extracting clip {segment_id}[/red]")
                raise
            except subprocess.CalledProcessError as e:
                console.print(f"[red]❌ FFmpeg error for clip {segment_id}: {e.stderr}[/red]")
                raise
            except Exception as e:
                console.print(f"[red]❌ Error extracting clip {segment_id}: {e}[/red]")
                raise
            
            progress.update(task, advance=1)
    
    # 4. 保存切割信息
    clips_info = {
        'metadata': {
            'total_clips': len(clips_list),
            'source_video': video_file,
            'output_directory': output_dir,
            'extraction_method': 'ffmpeg_copy' if use_copy_codec else 'ffmpeg_transcode'
        },
        'clips': clips_metadata
    }
    
    clips_info_file = os.path.join(output_dir, "clips_info.json")
    with open(clips_info_file, 'w', encoding='utf-8') as f:
        json.dump(clips_info, f, ensure_ascii=False, indent=2)
    
    # 5. 生成汇总信息
    total_clips_duration = sum(c['duration'] for c in clips_metadata)
    
    console.print(Panel(
        f"[bold green]✅ Video Clips Extraction Complete[/bold green]\n"
        f"Total Clips: {len(clips_list)}\n"
        f"Total Duration: {total_clips_duration:.1f}s\n"
        f"Output Directory: {output_dir}\n"
        f"Info File: {clips_info_file}"
    ))
    
    return clips_list


def merge_all_video_clips(
    clips_dir: str = "output/batch_summary/clips",
    output_file: str = "output/batch_summary/merged_clips.mp4"
) -> str:
    """
    合并所有视频片段成一个文件
    
    Args:
        clips_dir: 包含所有 clip_*.mp4 文件的目录
        output_file: 输出的合并视频文件路径
    
    Returns:
        output_file: 生成的合并视频路径
    
    工作流程:
        1. 查找所有 clip_*.mp4 文件（按编号排序）
        2. 创建 concat 列表文件（ffmpeg 的 concat demuxer 要求）
        3. 使用 ffmpeg concat demuxer 合并（保持质量，快速）
        4. 生成最终的合并视频
    """
    
    console.print(Panel(f"[bold cyan]🎬 Merging video clips[/bold cyan]"))
    
    # 1. 查找所有 clip 文件
    clip_pattern = "clip_*.mp4"
    clip_files = sorted(Path(clips_dir).glob(clip_pattern))
    
    if not clip_files:
        console.print(f"[red]❌ No clip files found in {clips_dir}[/red]")
        raise FileNotFoundError(f"No {clip_pattern} files in {clips_dir}")
    
    console.print(f"[cyan]Found {len(clip_files)} clips to merge[/cyan]")
    
    # 2. 创建 ffmpeg concat 列表
    concat_list_file = os.path.join(clips_dir, "concat_list.txt")
    
    with open(concat_list_file, 'w', encoding='utf-8') as f:
        for clip_file in clip_files:
            # ffmpeg concat 格式: file '/path/to/file'
            f.write(f"file '{clip_file.absolute()}'\n")
    
    console.print(f"[cyan]Created concat list: {concat_list_file}[/cyan]")
    
    # 3. 使用 ffmpeg 的 concat demuxer 合并（无转码，速度最快）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',  # 允许绝对路径
        '-i', concat_list_file,
        '-c', 'copy',  # 直接复制，不转码
        '-y',
        output_file
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        # 验证输出文件
        if os.path.exists(output_file):
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            total_duration = sum_clip_durations(clip_files)
            
            console.print(Panel(
                f"[bold green]✅ Clips merged successfully[/bold green]\n"
                f"Output: {output_file}\n"
                f"Size: {file_size_mb:.1f} MB\n"
                f"Total Duration: {total_duration:.1f}s"
            ))
        else:
            raise FileNotFoundError("FFmpeg merge did not create output file")
    
    except subprocess.TimeoutExpired:
        console.print("[red]❌ Merge operation timed out[/red]")
        raise
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ FFmpeg merge error: {e.stderr}[/red]")
        raise
    
    return output_file


def sum_clip_durations(clip_files: List[Path]) -> float:
    """计算所有片段的总时长"""
    # 这是一个简化版本，实际可以用 ffprobe 获取准确时长
    # 这里假设从 clips_info.json 读取
    try:
        clips_info_file = clip_files[0].parent / "clips_info.json"
        if clips_info_file.exists():
            with open(clips_info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return sum(c['duration'] for c in data.get('clips', []))
    except:
        pass
    return 0.0


if __name__ == "__main__":
    # 测试：读取 key_segments.json 并提取视频片段
    try:
        with open("output/batch_summary/key_segments.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            key_segments = data['key_segments']
        
        clips = extract_video_clips_by_segments(key_segments)
        
        console.print("\n[bold cyan]Extraction Summary:[/bold cyan]")
        for clip in clips:
            console.print(f"  ✓ {clip}")
    
    except FileNotFoundError:
        console.print("[red]Error: Run _6_5_identify_key_segments first[/red]")
