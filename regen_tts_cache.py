"""
regen_tts_cache.py -- re-synthesize every phrase in tts_cache/manifest.json
with the CURRENT Piper voice. Run it by hand after changing the voice:

    python regen_tts_cache.py
    python regen_tts_cache.py --backup            # keep a copy of the old cache first
    python regen_tts_cache.py --dry-run           # just list what would be regenerated
    python regen_tts_cache.py --model piper/voices/new_voice.onnx

What it does:
  1. Loads the voice the same way ai_stream.py does (piper/voices/, first
     .onnx alphabetically, unless you pass --model).
  2. Reads manifest.json and takes every entry's text + object pattern.
  3. Synthesizes ALL phrases first (in memory). If any single phrase fails,
     nothing on disk is touched, so you never end up with a half-old,
     half-new cache.
  4. Only then overwrites the .wav files and rewrites manifest.json
     (atomically), keeping each entry's text / file / objects unchanged.

Stop ai_stream.py before running this (it holds the cache in memory and
could write entries in the old voice while you regenerate).
Put this file next to pi_audio.py.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time

from pi_audio import AudioError, AudioOutput

PIPER_DIR = "./piper"
TTS_CACHE_DIR = "./tts_cache"


def file_name_for(text: str) -> str:
    # Same naming scheme as AudioOutput._save_to_disk
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20] + ".wav"


def main():
    p = argparse.ArgumentParser(description="Regenerate the TTS cache with the current Piper voice")
    p.add_argument("--piper-dir", default=PIPER_DIR, help="folder containing voices/")
    p.add_argument("--model", default=None, help="explicit path to a .onnx voice (overrides --piper-dir)")
    p.add_argument("--cache-dir", default=TTS_CACHE_DIR, help="TTS cache folder")
    p.add_argument("--backup", action="store_true", help="copy the cache folder to <cache>_backup_<time> first")
    p.add_argument("--dry-run", action="store_true", help="list phrases, change nothing")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)

    manifest_path = os.path.join(args.cache_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        sys.exit(f"No manifest at {manifest_path} -- nothing to regenerate.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # De-duplicate by text, keeping the first entry (and any objects info).
    entries = {}
    for e in manifest:
        entries.setdefault(e["text"], e)
    texts = list(entries)
    print(f"Found {len(texts)} phrase(s) in {manifest_path}")

    if args.dry_run:
        for t in texts:
            print(f"  - {t!r}")
        print("\nDry run: nothing changed.")
        return

    # cache_dir=None on purpose: don't load the old audio into this instance.
    # We only use it for its loaded voice.
    try:
        if args.model:
            audio = AudioOutput(model_path=args.model)
        else:
            audio = AudioOutput(piper_dir=args.piper_dir)
    except AudioError as e:
        sys.exit(f"Could not load the voice: {e}")
    print(f"Using voice: {audio._model_path}\n")

    # Phase 1: synthesize everything in memory.
    new_audio = {}
    t_all = time.monotonic()
    for i, text in enumerate(texts, 1):
        t0 = time.monotonic()
        try:
            new_audio[text] = audio._synthesize_to_bytes(text)
        except Exception as e:
            sys.exit(f"\nFailed on {text!r}: {e}\nNo files were changed.")
        print(f"[{i}/{len(texts)}] {time.monotonic() - t0:.2f}s  {text!r}")

    # Phase 2: everything worked -- optional backup, then write to disk.
    if args.backup:
        base = os.path.normpath(args.cache_dir)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{base}_backup_{stamp}"
        shutil.copytree(args.cache_dir, backup_dir)
        print(f"\nBackup saved to {backup_dir}")

    new_manifest = []
    for text, wav_bytes in new_audio.items():
        e = entries[text]
        file_name = e.get("file") or file_name_for(text)
        with open(os.path.join(args.cache_dir, file_name), "wb") as f:
            f.write(wav_bytes)
        new_manifest.append({"text": text, "file": file_name, "objects": e.get("objects")})

    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)

    print(f"\nDone: regenerated {len(new_manifest)} phrase(s) in {time.monotonic() - t_all:.1f}s.")
    print("You can start ai_stream.py again.")


if __name__ == "__main__":
    main()