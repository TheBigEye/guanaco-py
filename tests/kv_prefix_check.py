"""KV prefix-reuse regression check.

Detects the "exact-prefix" duplication bug from guanaco-py 0.5.1-0.5.4 where
Llama.generate() re-evaluated the whole prompt on top of the KV cache on
every ordinary chat turn, doubling the cached context each time and making
long sessions slower and slower.

Run after building or installing a wheel:

    python examples/kv_prefix_check.py path/to/model.gguf

It runs two chat turns and applies the bookkeeping invariant that must hold
for a healthy backend:

    n_tokens  ~=  (last prompt tokens) + (all completion tokens)

On a buggy build the second turn overflows that bound by roughly the size
of an entire turn and this script exits 1. On a fixed build it exits 0.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to a GGUF model file")
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    from llama_cpp import Llama

    print(f"loading {args.model} ...")
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_batch=512,
        verbose=False,
    )

    system = "You are a helpful assistant. " * 40  # give the prefix real size
    turn1 = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Say hello and introduce yourself briefly."},
    ]
    turn2 = turn1 + [
        {"role": "assistant", "content": ""},  # replaced after turn 1
        {"role": "user", "content": "Now tell me a short joke."},
    ]

    completion_total = 0

    out1 = llm.create_chat_completion(messages=turn1, max_tokens=48)
    completion_total += out1["usage"]["completion_tokens"]
    kv1 = llm.n_tokens
    exp1 = out1["usage"]["prompt_tokens"] + completion_total
    print(f"turn 1: kv={kv1}  expected~{exp1}")

    turn2[2]["content"] = out1["choices"][0]["message"]["content"]
    out2 = llm.create_chat_completion(messages=turn2, max_tokens=48)
    completion_total += out2["usage"]["completion_tokens"]
    kv2 = llm.n_tokens
    exp2 = out2["usage"]["prompt_tokens"] + completion_total
    print(f"turn 2: kv={kv2}  expected~{exp2}")

    # Allow a small margin for the generator's N-1 rollback bookkeeping.
    MARGIN = 24
    if kv2 > exp2 + MARGIN:
        surplus = kv2 - exp2
        print(
            f"\nFAIL: KV cache holds {surplus} more tokens than the turn "
            "accounts for. The prompt was re-evaluated on top of the cache: "
            "this build has the exact-prefix duplication regression "
            "(fixed in guanaco-py 0.5.5)."
        )
        return 1

    # Also verify turn 2 did not pay a full prefill: only the genuinely new
    # suffix should have been evaluated. A buggy build evaluates the full
    # prompt again, which shows up in perf counters when verbose=True; the
    # ledger check above is authoritative, so this is informational.
    print("\nOK: multi-turn KV prefix reuse is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
