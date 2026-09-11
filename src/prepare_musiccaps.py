"""Create leak-free deterministic MusicCaps splits after preprocessing.

The split is over clip IDs because MusicCaps has no artist identity.  Only
clips that have both a downloaded waveform and a built graph are included.
"""
import json
import os
import random

from utils import load_config


def main():
    cfg = load_config()
    graph_dir = os.path.join(cfg["paths"]["processed_dir"], "graphs")
    audio_dir = os.path.join(cfg["paths"]["raw_dir"], "musiccaps", "audio")
    ids = sorted(
        os.path.splitext(name)[0] for name in os.listdir(graph_dir)
        if name.endswith(".pt") and os.path.exists(os.path.join(audio_dir, os.path.splitext(name)[0] + ".wav"))
    )
    rng = random.Random(cfg["datasets"]["seed"])
    rng.shuffle(ids)
    n = len(ids)
    train_end = int(n * cfg["datasets"]["train_fraction"])
    val_end = train_end + int(n * cfg["datasets"]["val_fraction"])
    splits = {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}
    out_dir = cfg["paths"]["splits_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for name, values in splits.items():
        with open(os.path.join(out_dir, f"musiccaps_{name}.json"), "w") as handle:
            json.dump(values, handle, indent=2)
    print({name: len(values) for name, values in splits.items()})


if __name__ == "__main__":
    main()
