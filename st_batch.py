import math
import os
import re
import sys
import hashlib
import subprocess
import platform
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st
from yt_dlp import YoutubeDL

from batch.utils import batch_processor


# Match `st.py` behavior so imports and PATH work reliably under Streamlit.
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir
sys.path.append(current_dir)


TASKS_PATH = "batch/tasks_setting.xlsx"


def _to_int(v, default=0):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return int(v)
    except Exception:
        return default


def _format_duration(seconds: int | None) -> str:
    s = _to_int(seconds, 0)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _parse_upload_date(upload_date: str | None) -> str:
    if not upload_date:
        return ""
    # yt-dlp uses YYYYMMDD
    if re.fullmatch(r"\d{8}", str(upload_date)):
        try:
            dt = datetime.strptime(str(upload_date), "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return str(upload_date)
    return str(upload_date)


def _match_score(keyword: str, title: str) -> float:
    k = (keyword or "").strip().lower()
    t = (title or "").strip().lower()
    if not k or not t:
        return 0.0
    ratio = SequenceMatcher(None, k, t).ratio()
    words = [w for w in re.split(r"\s+", k) if w]
    overlap = sum(1 for w in words if w in t) / max(len(words), 1)
    return 0.7 * ratio + 0.3 * overlap


@st.cache_data(show_spinner=False, ttl=600)
def youtube_search(keyword: str, max_results: int = 10) -> pd.DataFrame:
    keyword = (keyword or "").strip()
    if not keyword:
        return pd.DataFrame()

    # Use yt-dlp search. Likes may be missing for some videos.
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "nocheckcertificate": True,
        "extract_flat": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{keyword}", download=False)

    rows = []
    for e in (info or {}).get("entries", []) or []:
        if not e:
            continue
        title = e.get("title") or ""
        url = e.get("webpage_url") or (f"https://www.youtube.com/watch?v={e.get('id')}" if e.get("id") else "")
        rows.append(
            {
                "Select": True,
                "Title": title,
                "Duration": _format_duration(e.get("duration")),
                "Likes": _to_int(e.get("like_count"), 0),
                "Views": _to_int(e.get("view_count"), 0),
                "Upload Date": _parse_upload_date(e.get("upload_date")),
                "URL": url,
                "_upload_raw": _to_int(e.get("timestamp"), 0),
                "_duration_raw": _to_int(e.get("duration"), 0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort (desc): upload time, views, duration, likes
    df = df.sort_values(by=["_upload_raw", "Views", "_duration_raw", "Likes"], ascending=[False, False, False, False])
    df = df.drop(columns=["_upload_raw", "_duration_raw"])
    return df.reset_index(drop=True)


def ensure_tasks_file() -> None:
    batch_processor.load_tasks_df(TASKS_PATH)


def append_selected_to_tasks(selected_df: pd.DataFrame) -> None:
    urls = [u for u in selected_df.get("URL", []).tolist() if isinstance(u, str) and u.strip()]
    batch_processor.append_tasks(urls, TASKS_PATH)


def main():
    st.set_page_config(page_title="VideoLingo Batch", layout="wide")
    st.title("VideoLingo — Batch UI")

    ensure_tasks_file()

    st.subheader("a. 搜索")
    keyword = st.text_input("关键词", value=st.session_state.get("keyword", ""))
    col1, col2 = st.columns([1, 5])
    with col1:
        searching = bool(st.session_state.get("searching", False))
        do_search = st.button(
            "搜索中" if searching else "搜索",
            use_container_width=True,
            disabled=searching,
        )
    with col2:
        if searching:
            st.caption("搜索中")
        else:
            st.caption("最多展示 10 个视频。")

    if do_search:
        st.session_state["keyword"] = keyword
        st.session_state["searching"] = True
        with st.spinner("搜索中..."):
            st.session_state["search_df"] = youtube_search(keyword, max_results=10)
        st.session_state["searching"] = False

    st.subheader("b. 选择视频")
    search_df = st.session_state.get("search_df", pd.DataFrame())
    if search_df is None or search_df.empty:
        st.info("请先搜索关键词。")
    else:
        edited = st.data_editor(
            search_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Select": st.column_config.CheckboxColumn("选择", default=True),
                "Title": st.column_config.TextColumn("视频名", width="large"),
                "Duration": st.column_config.TextColumn("时长"),
                "Likes": st.column_config.NumberColumn("点赞数"),
                "Views": st.column_config.NumberColumn("播放量"),
                "Upload Date": st.column_config.TextColumn("上传时间"),
                "URL": st.column_config.TextColumn("URL", width="large"),
            },
            disabled=["Title", "Duration", "Likes", "Views", "Upload Date", "URL"],
        )

        selected = edited[edited["Select"] == True].copy()  # noqa: E712

        if st.button("添加选中视频到任务列表", use_container_width=True):
            append_selected_to_tasks(selected)
            st.success(f"已添加 {len(selected)} 条（自动去重）。")
            st.session_state["selected_urls"] = selected["URL"].tolist()

    st.subheader("c. 开始处理视频")
    selected_urls = st.session_state.get("selected_urls", [])
    if not selected_urls:
        st.info("请先在上一步添加视频到 tasks_setting.xlsx（会记录本次选中列表）。")
        return

    status_placeholder = st.empty()

    def _stable_key(prefix: str, video_file: str) -> str:
        h = hashlib.md5((video_file or "").encode("utf-8"), usedforsecurity=False).hexdigest()  # noqa: S324
        return f"{prefix}_{h}"

    def _open_in_system(path: Path) -> None:
        p = path.resolve()
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", str(p)])
            elif platform.system() == "Windows":
                os.startfile(str(p))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            st.error(f"无法打开文件: {e}")

    def _open_text_in_system(path: Path) -> None:
        p = path.resolve()
        try:
            if platform.system() == "Darwin":
                # -t forces opening with the default text editor.
                subprocess.Popen(["open", "-t", str(p)])
            elif platform.system() == "Windows":
                subprocess.Popen(["notepad.exe", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            st.error(f"无法用文本编辑器打开文件: {e}")

    def _task_dir_from_row(row: pd.Series) -> Path | None:
        output_dir = row.get("Output Dir")
        if not isinstance(output_dir, str) or not output_dir.strip():
            # Fallback: resolve by source_url.txt (written during download/copy)
            video_file = str(row.get("Video File") or "").strip()
            if not video_file:
                return None
            base = Path("batch/output")
            if base.exists():
                for child in base.iterdir():
                    if not child.is_dir() or child.name.startswith(".") or child.name == "ERROR":
                        continue
                    marker = child / "source_url.txt"
                    if not marker.exists():
                        continue
                    try:
                        v = marker.read_text(encoding="utf-8").strip()
                    except Exception:
                        v = marker.read_text(encoding="utf-8", errors="ignore").strip()
                    if v == video_file:
                        return child
            return None
        p = Path(output_dir)
        return p if p.exists() else None

    def _summary_video_path(task_dir: Path) -> Path:
        # Spec path: batch_summary/merged_clips.mp4
        p = task_dir / "batch_summary" / "merged_clips.mp4"
        if p.exists():
            return p
        # Fallback if only dubbed version exists.
        p2 = task_dir / "batch_summary" / "merged_clips_dub.mp4"
        return p2

    def _render_status():
        df_tasks = batch_processor.load_tasks_df(TASKS_PATH)
        df_show = df_tasks[df_tasks["Video File"].isin(selected_urls)].copy()
        if df_show.empty:
            status_placeholder.info("任务列表为空。")
            return

        with status_placeholder.container():
            header = st.columns([3, 1.2, 2.4, 2.4, 3.0])
            header[0].markdown("**Video File**")
            header[1].markdown("**Status**")
            header[2].markdown("**中英文字幕视频**")
            header[3].markdown("**摘要视频**")
            header[4].markdown("**中文字幕**")

            for _, row in df_show.iterrows():
                video_file = str(row.get("Video File") or "")
                status = str(row.get("Status") or "")
                task_dir = _task_dir_from_row(row)

                cols = st.columns([3, 1.2, 2.4, 2.4, 3.0])
                cols[0].write(video_file)
                cols[1].write(status)

                if task_dir and ("Done" in status):
                    sub_video = task_dir / "output_sub.mp4"
                    if sub_video.exists():
                        if cols[2].button("打开", key=_stable_key("open_sub", video_file), use_container_width=True):
                            _open_in_system(sub_video)
                    else:
                        cols[2].write("-")

                    summary_video = _summary_video_path(task_dir)
                    if summary_video.exists():
                        if cols[3].button("打开", key=_stable_key("open_summary", video_file), use_container_width=True):
                            _open_in_system(summary_video)
                    else:
                        cols[3].write("-")

                    trans_srt = task_dir / "trans.srt"
                    if trans_srt.exists():
                        if cols[4].button("打开", key=_stable_key("open_srt", video_file), use_container_width=True):
                            _open_text_in_system(trans_srt)
                    else:
                        cols[4].write("-")
                else:
                    cols[2].write("-")
                    cols[3].write("-")
                    cols[4].write("-")

    # Always render the current results table (important after a rerun).
    _render_status()

    def progress_cb(video_url: str, status: str):
        # Update UI on each status change
        _render_status()

    processing = bool(st.session_state.get("processing", False))
    start_clicked = st.button(
        "开始处理",
        type="primary",
        use_container_width=True,
        key="start_processing",
        disabled=processing,
    )

    if start_clicked:
        st.session_state["processing"] = True
        try:
            # Ensure tasks exist for these URLs
            batch_processor.append_tasks(selected_urls, TASKS_PATH)
            _render_status()

            batch_processor.process_selected_videos(
                selected_urls,
                TASKS_PATH,
                progress_cb=progress_cb,
                merge_summaries=False,
            )
            _render_status()
            st.success("处理完成。")
        finally:
            st.session_state["processing"] = False
            st.rerun()

    st.subheader("d. 合并多个摘要视频")
    merged_placeholder = st.empty()
    if st.button("合并多个摘要视频", use_container_width=True):
        with st.spinner("合并中..."):
            merged = batch_processor.merge_successful_summaries()
        if merged and os.path.exists(merged):
            merged_placeholder.video(merged)
        else:
            merged_placeholder.info("未找到可合并的摘要视频或合并失败。")


if __name__ == "__main__":
    main()
