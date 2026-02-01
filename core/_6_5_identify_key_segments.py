import pandas as pd
import json
import os
from typing import List, Dict, Any
from core.utils import *
from core.utils.ask_gpt import ask_gpt
from core.utils.models import *
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def identify_key_segments_from_subtitles(
    df_trans_time: pd.DataFrame = None,
    output_dir: str = "output/batch_summary",
    max_segments: int = 4,
    max_coverage_ratio: float = 0.30,
    min_importance_score: float = 0.80
) -> List[Dict[str, Any]]:
    """
    识别关键段 - 基于最终的中文字幕（已对齐时间戳）
    
    执行点: _6_gen_sub.align_timestamp_main() ✓ 完成 → 本函数 ⭐
    
    Args:
        df_trans_time: 包含时间戳的字幕 DataFrame
                      columns: Source, Translation, timestamp, duration
                      来自 _6_gen_sub.align_timestamp()
        output_dir: 输出目录
        max_segments: 最多识别的关键段数
        max_coverage_ratio: 关键段占总时长的最大比例（0.3 = 30%）
        min_importance_score: 最低重要度分数（0.0-1.0）
    
    Returns:
        key_segments 列表: 每个关键段包含 start_time, end_time, importance, reason 等信息
    
    ⚠️ 重要提醒：
        此时的字幕已经精确对齐，每一行对应的时间戳都是准确的。
        我们基于最终要显示给用户的中文字幕来识别关键段。
    """
    
    # 1. 如果没有传入 df_trans_time，从文件读取
    if df_trans_time is None:
        try:
            df_trans_time = pd.read_excel("output/log/trans_time_with_timestamp.xlsx")
        except FileNotFoundError:
            console.print("[red]❌ Error: trans_time_with_timestamp.xlsx not found[/red]")
            console.print("[yellow]Please ensure _6_gen_sub.align_timestamp_main() has been executed[/yellow]")
            raise
    
    console.print(Panel(f"[bold cyan]🔍 Identifying key segments from {len(df_trans_time)} subtitle lines[/bold cyan]"))
    
    # 2. 构建字幕上下文（包含时间戳和中文文本）
    subtitles_context = "\n".join([
        f"[{row['timestamp']}] {row['Translation']}"
        for idx, row in df_trans_time.iterrows()
    ])
    
    # 3. 计算视频总时长
    # 从最后一行的 timestamp 提取结束时间
    last_timestamp = df_trans_time.iloc[-1]['timestamp']
    # 格式: "00:01:23,456 --> 00:02:15,789"
    end_time_str = last_timestamp.split(' --> ')[1]  # "00:02:15,789"
    video_total_seconds = parse_srt_time(end_time_str)
    
    # 4. 使用 LLM 识别关键段
    prompt = f"""
请仔细分析以下带有时间戳的中文字幕，识别最重要的 {max_segments} 个关键段落。

分析标准:
1. 包含最核心的信息（新闻、发现、结论、重大进展等）
2. 总时长不超过视频总长的 {int(max_coverage_ratio*100)}%
3. 内容连贯完整，易于独立观看
4. 每个关键段应该是完整的思想单元

视频总时长: {video_total_seconds:.1f} 秒
最大关键段总时长: {video_total_seconds * max_coverage_ratio:.1f} 秒

字幕及时间戳（格式: HH:MM:SS,mmm --> HH:MM:SS,mmm）:
{subtitles_context}

请返回 JSON 格式的结果（注意: start_time 和 end_time 必须是字幕中出现的时间戳）:
{{
    "key_segments": [
        {{
            "segment_id": 1,
            "start_time": "00:01:23,456",
            "end_time": "00:02:15,789",
            "importance": 0.95,
            "reason": "介绍重大突破发现，是整个视频的核心内容",
            "zh_summary": "科学家发现了新的粒子"
        }},
        ...最多 {max_segments} 个
    ],
    "total_analysis": "整体内容分析"
}}

要求:
- start_time 和 end_time 必须是字幕时间戳中的实际值
- importance 是 0.0-1.0 的浮点数
- 关键段要能覆盖视频最重要的内容
"""
    
    result = ask_gpt(prompt, resp_type='json', log_title='key_segments_from_subtitles')
    
    # 5. 验证和处理结果
    validated_segments = []
    total_duration = 0
    
    for seg in result.get('key_segments', []):
        try:
            # 解析时间戳
            start_seconds = parse_srt_time(seg['start_time'])
            end_seconds = parse_srt_time(seg['end_time'])
            duration = end_seconds - start_seconds
            
            # 验证
            if start_seconds >= end_seconds:
                console.print(f"[yellow]⚠️ Skipping segment: start >= end[/yellow]")
                continue
            
            if end_seconds > video_total_seconds:
                console.print(f"[yellow]⚠️ Skipping segment: end time beyond video duration[/yellow]")
                continue
            
            if seg.get('importance', 0) < min_importance_score:
                console.print(f"[yellow]⚠️ Skipping segment: importance {seg['importance']} < {min_importance_score}[/yellow]")
                continue
            
            # 添加到验证列表
            validated_segments.append({
                'segment_id': seg['segment_id'],
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'start_seconds': start_seconds,
                'end_seconds': end_seconds,
                'duration': duration,
                'importance': seg['importance'],
                'reason': seg['reason'],
                'zh_summary': seg.get('zh_summary', '')
            })
            
            total_duration += duration
            
            console.print(f"[green]✓[/green] Segment {seg['segment_id']}: "
                        f"{seg['start_time']} → {seg['end_time']} "
                        f"({duration:.1f}s, importance: {seg['importance']:.2f})")
        
        except Exception as e:
            console.print(f"[red]❌ Error processing segment: {e}[/red]")
            continue
    
    # 6. 检查覆盖率
    coverage_ratio = total_duration / video_total_seconds if video_total_seconds > 0 else 0
    
    console.print(Panel(
        f"[bold green]✅ Key Segments Identified[/bold green]\n"
        f"Count: {len(validated_segments)}\n"
        f"Total Duration: {total_duration:.1f}s / {video_total_seconds:.1f}s\n"
        f"Coverage Ratio: {coverage_ratio*100:.1f}%"
    ))
    
    # 7. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'metadata': {
            'video_total_duration': video_total_seconds,
            'total_key_duration': total_duration,
            'coverage_ratio': coverage_ratio,
            'segment_count': len(validated_segments)
        },
        'key_segments': validated_segments
    }
    
    with open(os.path.join(output_dir, "key_segments.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 8. 生成 Excel 汇总表
    if validated_segments:
        df_segments = pd.DataFrame(validated_segments)
        df_segments.to_excel(os.path.join(output_dir, "key_segments.xlsx"), index=False)
        console.print(f"[green]💾 Saved to {output_dir}/key_segments.json and .xlsx[/green]")
    
    return validated_segments


def parse_srt_time(time_str: str) -> float:
    """
    将 SRT 时间格式转换为秒数
    
    输入: "00:01:23,456"
    输出: 83.456 (秒)
    """
    parts = time_str.replace(',', '.').split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def format_seconds_to_srt(seconds: float) -> str:
    """
    将秒数转换为 SRT 时间格式
    
    输入: 83.456
    输出: "00:01:23,456"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"


if __name__ == "__main__":
    # 测试：读取 trans_time_with_timestamp.xlsx 并识别关键段
    try:
        df = pd.read_excel("output/log/trans_time_with_timestamp.xlsx")
        key_segments = identify_key_segments_from_subtitles(df)
        
        console.print("\n[bold cyan]Key Segments Summary:[/bold cyan]")
        for seg in key_segments:
            console.print(f"  Segment {seg['segment_id']}: {seg['start_time']} - {seg['end_time']}")
            console.print(f"    Reason: {seg['reason']}")
            console.print(f"    Summary: {seg['zh_summary']}\n")
    
    except FileNotFoundError:
        console.print("[red]Error: Run _6_gen_sub first[/red]")
