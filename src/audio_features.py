"""
audio_features.py — Preprocessing pipeline, steps 1-2 of the spec:
  1. Resample to 22,050 Hz; extract log-mel (128 bins) or chroma (12 bins); normalize per track.
  2. Split each track into fixed windows (5-10s) using librosa.

Produces one .npy per track under data/processed/features/{track_id}.npy with
shape [n_segments, n_frames_per_segment, n_bins]. graph_builder.py consumes
these arrays to build both chord-transition and segment-similarity graphs.

Assumes raw audio files live under the folders documented at the top of
config.yaml (data/raw/<dataset_name>/...).
"""
import os
import numpy as np
import librosa
from tqdm import tqdm

from utils import load_config


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    """Load and resample a single audio file to a mono waveform."""
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    return y


def extract_log_mel(y: np.ndarray, cfg: dict) -> np.ndarray:
    """log-mel spectrogram, shape [n_frames, n_mels]."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg["audio"]["sample_rate"],
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
        n_mels=cfg["audio"]["n_mels"],
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T  # [n_frames, n_mels]


def extract_chroma(y: np.ndarray, cfg: dict) -> np.ndarray:
    """Chroma (pitch-class) features, shape [n_frames, 12]. Used for chord graphs."""
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=cfg["audio"]["sample_rate"],
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
    )
    return chroma.T  # [n_frames, 12]


def normalize(feat: np.ndarray) -> np.ndarray:
    """Per-track z-score normalization (mean 0, std 1 across the time axis)."""
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + 1e-8
    return (feat - mean) / std


def segment_track(feat: np.ndarray, sr: int, hop_length: int, segment_seconds: float) -> np.ndarray:
    """
    Split a [n_frames, n_bins] feature matrix into fixed-length windows.
    Returns [n_segments, frames_per_segment, n_bins]. Drops a trailing partial
    segment shorter than half a window.
    """
    frames_per_second = sr / hop_length
    frames_per_segment = int(round(frames_per_second * segment_seconds))
    n_total = feat.shape[0]
    n_segments = n_total // frames_per_segment
    if n_total % frames_per_segment > frames_per_segment // 2:
        # pad the last partial segment with zeros instead of dropping it
        pad_len = frames_per_segment - (n_total % frames_per_segment)
        feat = np.pad(feat, ((0, pad_len), (0, 0)), mode="constant")
        n_segments += 1
    trimmed = feat[: n_segments * frames_per_segment]
    return trimmed.reshape(n_segments, frames_per_segment, feat.shape[1])


def process_track(path: str, cfg: dict) -> dict:
    """
    Full per-track pipeline: load -> resample -> extract both feature types
    -> normalize -> segment. Returns a dict with both feature streams because
    chroma feeds chord graphs while log-mel feeds segment-similarity graphs
    and the CNN baseline (Task 2's comparison point).
    """
    y = load_audio(path, cfg["audio"]["sample_rate"])

    mel = normalize(extract_log_mel(y, cfg))
    chroma = normalize(extract_chroma(y, cfg))

    mel_segments = segment_track(
        mel, cfg["audio"]["sample_rate"], cfg["audio"]["hop_length"], cfg["audio"]["segment_seconds"]
    )
    chroma_segments = segment_track(
        chroma, cfg["audio"]["sample_rate"], cfg["audio"]["hop_length"], cfg["audio"]["segment_seconds"]
    )
    return {"mel": mel_segments, "chroma": chroma_segments}


def build_feature_cache(dataset_dir: str, out_dir: str, cfg: dict, extensions=(".mp3", ".wav")) -> None:
    """
    Walk `dataset_dir` for audio files, extract features for each, and save
    to `out_dir/{track_id}.npz`. track_id = filename stem.
    """
    os.makedirs(out_dir, exist_ok=True)
    audio_paths = []
    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            if fn.lower().endswith(extensions):
                audio_paths.append(os.path.join(root, fn))

    # Keep a deterministic development subset by default.  This makes the
    # complete assignment runnable on a laptop; set datasets.max_tracks: null
    # in config.yaml for the full corpus.
    audio_paths.sort()
    limit = cfg["datasets"].get("max_tracks")
    if limit:
        audio_paths = audio_paths[:int(limit)]
    print(f"Processing {len(audio_paths)} audio files under {dataset_dir}")
    for path in tqdm(audio_paths, desc="Extracting features"):
        track_id = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{track_id}.npz")
        if os.path.exists(out_path):
            continue  # resumable: skip already-processed tracks
        try:
            feats = process_track(path, cfg)
            np.savez_compressed(out_path, mel=feats["mel"], chroma=feats["chroma"])
        except Exception as e:
            print(f"  [skip] {track_id}: {e}")


if __name__ == "__main__":
    cfg = load_config()
    active = cfg["datasets"]["active_audio"]
    raw_dir = os.path.join(cfg["paths"]["raw_dir"], active)
    out_dir = os.path.join(cfg["paths"]["processed_dir"], "features")
    build_feature_cache(raw_dir, out_dir, cfg)
