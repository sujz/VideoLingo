import platform
import subprocess

import cv2
import numpy as np
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'


def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE

    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")
        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()
        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    # 获取视频分辨率
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    font_name, font_size, font_color, outline_color, outline_width, back_color = get_subtitle_font_config()
    subtitle_filter = (
        f"subtitles={DUB_SUB_FILE}:force_style='FontName={font_name},FontSize={font_size},PrimaryColour={font_color},"
        f"OutlineColour={outline_color},OutlineWidth={outline_width},BackColour={back_color},Alignment=2,MarginV=27,BorderStyle=4'"
    )

    cmd = [
        'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', normalized_dub_audio,
        '-filter_complex',
        f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
        f'{subtitle_filter}[v];'
        f'[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[a]'
    ]

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]'])

    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])

    subprocess.run(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")


def get_subtitle_font_config():
    font_name = load_key('subtitle_font.name')
    font_size = load_key('subtitle_font.size')
    font_color = load_key('subtitle_font.color')
    outline_color = load_key('subtitle_font.outline_color')
    outline_width = load_key('subtitle_font.outline_width')
    back_color = load_key('subtitle_font.back_color')
    return font_name, font_size, font_color, outline_color, outline_width, back_color

if __name__ == '__main__':
    merge_video_audio()
