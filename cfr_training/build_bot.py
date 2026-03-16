"""
build_bot.py
============
Run after training to produce a single self-contained bot.py.

Usage:
    python build_bot.py --strategy strategy.pkl --template bot_template.py --out bot.py

The script:
  1. Loads strategy.pkl
  2. Compresses it with zlib (level 9)
  3. Encodes as base64
  4. Replaces the sentinel line in bot_template.py with the embedded constant
  5. Writes the final bot.py — the only file you need to submit
"""

import argparse
import base64
import os
import pickle
import sys
import zlib


SENTINEL = "STRATEGY_DATA = None  # <-- build_bot.py replaces this line"


def embed_strategy(pkl_path: str, template_path: str, out_path: str):
    # ── Load strategy ─────────────────────────────────────────────────────────
    print(f"Loading strategy from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        strategy = pickle.load(f)
    print(f"  Infosets: {len(strategy):,}")

    # ── Compress and encode ───────────────────────────────────────────────────
    raw  = pickle.dumps(strategy, protocol=4)
    comp = zlib.compress(raw, level=9)
    b64  = base64.b64encode(comp).decode("ascii")

    raw_mb  = len(raw)  / 1e6
    comp_mb = len(comp) / 1e6
    b64_mb  = len(b64)  / 1e6
    print(f"  Raw:        {raw_mb:.2f} MB")
    print(f"  Compressed: {comp_mb:.2f} MB  ({len(raw)/len(comp):.1f}x ratio)")
    print(f"  Base64:     {b64_mb:.2f} MB  (this goes into the file)")

    # ── Load template ─────────────────────────────────────────────────────────
    print(f"\nLoading template from {template_path} ...")
    with open(template_path, "r") as f:
        template = f.read()

    if SENTINEL not in template:
        print(f"ERROR: sentinel line not found in template:")
        print(f"  Expected: {SENTINEL!r}")
        sys.exit(1)

    # ── Inject ────────────────────────────────────────────────────────────────
    # Write the constant as a multi-line string so editors don't choke on one
    # giant line. We chunk the b64 string into 80-char lines.
    chunk_size = 80
    chunks     = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]
    b64_literal = '(\n    "' + '"\n    "'.join(chunks) + '"\n)'
    replacement = f"STRATEGY_DATA = {b64_literal}"

    output = template.replace(SENTINEL, replacement)

    # ── Write ─────────────────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        f.write(output)

    final_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWrote {out_path}  ({final_mb:.2f} MB)")
    print("Done — submit this single file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="strategy.pkl",
                        help="path to strategy.pkl produced by cfr_train.py")
    parser.add_argument("--template", default="bot_template.py",
                        help="bot template file (contains the sentinel line)")
    parser.add_argument("--out",      default="bot.py",
                        help="output file path")
    args = parser.parse_args()

    embed_strategy(args.strategy, args.template, args.out)
