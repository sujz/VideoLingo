import os
import sys
import argparse
import shutil
import tempfile
import subprocess

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--dubbing', type=int, default=0)
    p.add_argument('--is_retry', type=int, default=0)
    p.add_argument('--source', default='')
    p.add_argument('--target', default='')
    p.add_argument('--index', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    # create temp config copy and set env var before importing core modules
    root = os.getcwd()
    orig_config = os.path.join(root, 'config.yaml')
    tmpdir = tempfile.mkdtemp(prefix='vl_config_')
    tmp_config = os.path.join(tmpdir, 'config.yaml')
    try:
        shutil.copy2(orig_config, tmp_config)
    except Exception:
        # if copy fails, still proceed with original config
        tmp_config = orig_config

    # set environment variable so core.utils.config_utils picks it up
    os.environ['VIDEO_LINGO_CONFIG_PATH'] = tmp_config

    # Now import modules (they will read config using tmp path)
    try:
        from batch.utils.video_processor import process_video
        # update language keys in config for this worker
        try:
            from core.utils.config_utils import update_key
            if args.source:
                update_key('whisper.language', args.source)
            if args.target:
                update_key('target_language', args.target)
        except Exception:
            pass

        is_retry = bool(args.is_retry)
        status, step, msg = process_video(args.video, args.dubbing, is_retry)
        if status:
            sys.exit(0)
        else:
            print(f"Worker failed: {step} - {msg}")
            sys.exit(2)
    except Exception as e:
        print(f"Worker exception: {e}")
        sys.exit(3)
    finally:
        # cleanup temp config
        try:
            if tmp_config != orig_config:
                os.remove(tmp_config)
                os.rmdir(tmpdir)
        except Exception:
            pass

if __name__ == '__main__':
    main()
