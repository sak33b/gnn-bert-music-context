"""
download_musiccaps.py — pulls the actual audio for MusicCaps.

MusicCaps is published as a CSV of (YouTube video ID, start_s, end_s,
caption, ...) — Google doesn't redistribute the audio itself, so this script
downloads each clip from YouTube directly and trims it to the labeled
window. Some videos will have been taken down since the dataset was
published; those are logged and skipped rather than stopping the run.

Requires (not installed by requirements.txt, since it's a one-time setup step):
    pip install yt-dlp
    ffmpeg installed and on PATH (yt-dlp shells out to it for trimming/conversion)

Usage:
    python src/download_musiccaps.py
    python src/download_musiccaps.py --limit 200          # test on a small subset first
    python src/download_musiccaps.py --workers 4           # parallel downloads
    python src/download_musiccaps.py --retry-failed        # only retry rows in failed_downloads.json
"""
import os
import json
import argparse
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from utils import load_config

YOUTUBE_URL_BASE = "https://www.youtube.com/watch?v="


def check_dependencies():
    """Fail fast with a clear message rather than 500 confusing per-row errors."""
    missing = [tool for tool in ("yt-dlp", "ffmpeg") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Missing required tool(s): {', '.join(missing)}. "
            f"Install with `pip install yt-dlp` and/or your OS package manager for ffmpeg."
        )


def download_clip(ytid: str, start_s: float, end_s: float, out_path: str,
                   sample_rate: int, num_attempts: int = 2) -> tuple:
    """
    Downloads and trims one clip. Returns (ytid, success: bool, message: str).
    Uses yt-dlp's --download-sections so only the needed window is fetched,
    not the full source video, and converts straight to wav at the project's
    sample rate so audio_features.py can load it without re-resampling.
    """
    if os.path.exists(out_path):
        return ytid, True, "already exists"

    section = f"*{start_s}-{end_s}"
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ar {sample_rate} -ac 1",
        "-f", "bestaudio/best",
        "--download-sections", section,
        "-o", out_path,
        f"{YOUTUBE_URL_BASE}{ytid}",
    ]

    last_err = ""
    for attempt in range(num_attempts):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            if os.path.exists(out_path):
                return ytid, True, "downloaded"
            last_err = "yt-dlp exited 0 but no output file was produced"
        except subprocess.CalledProcessError as e:
            last_err = e.stderr.strip().splitlines()[-1] if e.stderr else str(e)
        except subprocess.TimeoutExpired:
            last_err = "timed out"
    return ytid, False, last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Only download the first N rows (for a quick test run)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads. Keep modest to avoid YouTube rate-limiting")
    parser.add_argument("--retry-failed", action="store_true", help="Only retry rows previously logged in failed_downloads.json")
    args = parser.parse_args()

    check_dependencies()
    cfg = load_config(args.config)
    raw_dir = os.path.join(cfg["paths"]["raw_dir"], "musiccaps")
    csv_path = os.path.join(raw_dir, "musiccaps-public.csv")
    audio_dir = os.path.join(raw_dir, "audio")
    failed_log_path = os.path.join(raw_dir, "failed_downloads.json")
    os.makedirs(audio_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Fetch it first, e.g.:\n"
            f'  python -c "from datasets import load_dataset; '
            f"load_dataset('google/MusicCaps', split='train').to_csv('{csv_path}')\""
        )

    df = pd.read_csv(csv_path)

    if args.retry_failed:
        if not os.path.exists(failed_log_path):
            print("No failed_downloads.json found — nothing to retry.")
            return
        with open(failed_log_path) as f:
            failed_ids = set(json.load(f).keys())
        df = df[df["ytid"].isin(failed_ids)]
        print(f"Retrying {len(df)} previously failed downloads")

    if args.limit:
        df = df.head(args.limit)

    print(f"Downloading {len(df)} MusicCaps clips to {audio_dir}")
    print("Some will fail because the source video was removed from YouTube — that's expected, not a bug.")

    results, failures = [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_clip, row.ytid, row.start_s, row.end_s,
                os.path.join(audio_dir, f"{row.ytid}.wav"), cfg["audio"]["sample_rate"],
            ): row.ytid
            for row in df.itertuples()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            ytid, success, message = future.result()
            results.append(success)
            if not success:
                failures[ytid] = message

    with open(failed_log_path, "w") as f:
        json.dump(failures, f, indent=2)

    n_success = sum(results)
    print(f"\nDone: {n_success}/{len(results)} clips downloaded successfully.")
    if failures:
        print(f"{len(failures)} failed — logged to {failed_log_path}.")
        print("Re-run with --retry-failed later; some may come back if it was a transient rate-limit, not a takedown.")


if __name__ == "__main__":
    main()
