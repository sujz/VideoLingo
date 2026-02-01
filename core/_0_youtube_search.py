
from typing import List, Dict
import os
os.environ['YT_DLP_JS_RUNTIMES'] = 'node'
os.environ['YT_DLP_REMOTE_COMPONENTS'] = 'ejs:github'
import json
from rich.console import Console

console = Console()

def search_youtube(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search YouTube using yt-dlp's ytsearch extractor.

    Returns a list of metadata dicts with keys: id, url, title, uploader, duration, view_count, description
    Saves results to `output/log/search_results.json`.
    """
    import subprocess
    import shlex
    results = []
    query_str = f"ytsearch{max_results}:{query}"
    cmd = f"yt-dlp \"{query_str}\" --dump-json --js-runtimes node --remote-components ejs:github"
    try:
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
        # yt-dlp --dump-json outputs one JSON object per line (NDJSON)
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        for line in lines:
            try:
                entry = json.loads(line)
                if not entry or 'id' not in entry or entry.get('_type') == 'error':
                    continue
                item = {
                    'id': entry.get('id'),
                    'url': entry.get('webpage_url') or entry.get('url'),
                    'title': entry.get('title'),
                    'uploader': entry.get('uploader'),
                    'duration': entry.get('duration'),
                    'view_count': entry.get('view_count'),
                    'description': entry.get('description'),
                    'thumbnail': entry.get('thumbnail')
                }
                results.append(item)
            except Exception:
                continue
    except Exception as e:
        console.print(f"[red]YouTube search failed (subprocess): {e}[/red]")
        return results

    os.makedirs('output/log', exist_ok=True)
    out_file = os.path.join('output/log', 'search_results.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({'query': query, 'results': results}, f, ensure_ascii=False, indent=2)

    console.print(f"[green]Saved {len(results)} search results to {out_file}[/green]")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('query', help='Search query')
    parser.add_argument('--max', type=int, default=5, help='Max results')
    args = parser.parse_args()
    res = search_youtube(args.query, args.max)
    for i, r in enumerate(res, 1):
        console.print(f"{i}. {r['title']} ({r['duration']}s) - {r['url']}")
