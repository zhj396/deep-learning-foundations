# -*- coding: utf-8 -*-
"""Command-line translation: Portuguese -> English.

The "script" counterpart of the Flask service: same backends, same
tokenization, same dynamic batcher -- no web server. Useful for demos,
translating test splits, and torch-free edge boxes.

Usage:
    python translate.py -e "o gato esta aqui"
    python translate.py --input test.pt --output test.en
    cat sentences.txt | python translate.py

    [--model base] [--backend torch|onnx (env BACKEND fallback)]
    [--mode greedy|beam] [--beam-size 4] [--length-penalty 0.6] [--max-len 128]

File/stdin modes print one translation per line, skipping blank input
lines (blank lines produce no output line); -e prints just the
translation.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List


def _read_input(args) -> List[str]:
    if args.text is not None:
        return [args.text]
    if args.input is not None:
        return Path(args.input).read_text(encoding="utf-8").splitlines()
    if not sys.stdin.isatty():
        return sys.stdin.read().splitlines()
    return []


def main() -> None:
    # translations contain diacritics-free English but sources are Portuguese;
    # keep legacy Windows consoles from crashing on print
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Translate Portuguese to English with a trained checkpoint "
                    "(torch) or ONNX export (onnxruntime)."
    )
    parser.add_argument("-e", "--text", type=str, default=None, help="translate one sentence")
    parser.add_argument("--input", type=str, default=None,
                        help="input .txt file (one sentence per line)")
    parser.add_argument("--output", type=str, default=None,
                        help="write translations here instead of stdout")
    parser.add_argument("--model", type=str, default="base",
                        help="model name under the artifacts dir")
    parser.add_argument("--backend", type=str, default=os.environ.get("BACKEND", "torch"),
                        choices=["torch", "onnx"], help="serving backend (env BACKEND fallback)")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    lines = [line.strip() for line in _read_input(args)]
    if not lines:
        parser.error("Provide -e TEXT, --input FILE, or pipe text on stdin")
    lines = [line for line in lines if line]
    if not lines:
        parser.error("Input contains no non-empty lines")

    if args.mode == "beam" and args.backend == "onnx":
        parser.error("The onnx backend is greedy-only; use --backend torch for beam search")

    from backends import ARTIFACTS_DIR, DecodeParams, get_backend
    from batcher import DynamicBatcher, RequestItem
    from tokenization import Translator

    model_dir = ARTIFACTS_DIR / args.model
    translator = Translator(sp_model_path=model_dir / "spm_en_pt.model")
    backend = get_backend(args.backend)

    params = DecodeParams(
        mode=args.mode,
        max_len=args.max_len,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        **translator.decode_params(),
    )

    # same batching machinery as the service: one shot through the batcher
    batcher = DynamicBatcher(
        lambda arr: backend.translate(args.model, arr, params),
        pad_idx=translator.pad_idx,
        max_batch_size=args.batch_size,
        wait_timeout_s=600.0,
    )
    items = [RequestItem(translator.encode(line), line) for line in lines]

    print(f"[translate] backend={backend.name} model={args.model} mode={args.mode} "
          f"sentences={len(items)}", file=sys.stderr)
    t0 = time.time()
    batcher.submit(items)

    translations = []
    failures = 0
    for item in items:
        try:
            translations.append(translator.decode(batcher.wait(item)))
        except Exception as e:
            failures += 1
            translations.append("")
            print(f"[translate] FAILED: {item.text[:60]!r}: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"[translate] done in {elapsed:.2f}s ({len(items) / max(elapsed, 1e-9):.1f} sent/s, "
          f"{failures} failure(s))", file=sys.stderr)

    if args.output:
        Path(args.output).write_text("\n".join(translations) + "\n", encoding="utf-8")
        print(f"[translate] wrote {args.output}", file=sys.stderr)
    elif args.text is not None:
        print(translations[0])
    else:
        print("\n".join(translations))

    sys.exit(1 if failures == len(items) else 0)


if __name__ == "__main__":
    main()
