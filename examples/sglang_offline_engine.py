"""Run sglang's offline engine with causal-conv1d-cute serving the conv calls.

The engine starts its scheduler in a separate process, and sglang loads plugins there, so the
switch is the same environment variable that a server uses. It has to be set before the engine
is created.

    python examples/sglang_offline_engine.py --model-path Qwen/Qwen3.8-27B
"""

import argparse
import os


def main():
    """Generate a few greedy completions."""
    os.environ["CAUSAL_CONV1D_CUTE_SGLANG"] = "1"
    import sglang as sgl

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3.8-27B")
    a = ap.parse_args()
    engine = sgl.Engine(
        model_path=a.model_path,
        trust_remote_code=True,
        mem_fraction_static=0.85,
        chunked_prefill_size=2048,
    )
    prompts = ["The capital of France is", "A depthwise causal convolution is"]
    outputs = engine.generate(prompts, {"temperature": 0, "max_new_tokens": 32})
    for prompt, out in zip(prompts, outputs):
        print(repr(prompt), "->", repr(out["text"]))
    engine.shutdown()


if __name__ == "__main__":
    main()
