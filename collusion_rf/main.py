"""CLI: python -m collusion_rf.main {generate,train,evaluate,penalise,all} [--seed N] [--quick]."""
import argparse
import time

from .config import RunConfig


def main(argv: list[str] | None = None) -> None:
    """Run one pipeline stage, or all of them in order."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["generate", "train", "evaluate", "penalise", "all"])
    ap.add_argument("--seed", type=int, default=42, help="global seed (default 42)")
    ap.add_argument("--quick", action="store_true", help="small smoke-test sizes")
    ap.add_argument("--out", default="outputs", help="output directory")
    args = ap.parse_args(argv)
    cfg = RunConfig(seed=args.seed, out=args.out)
    if args.quick:
        cfg = cfg.as_quick()

    # imported lazily so `--help` stays fast
    from .dataset import generate
    from .evaluate import evaluate
    from .regulator import penalise
    from .train import train
    stages = {"generate": generate, "train": train, "evaluate": evaluate, "penalise": penalise}
    for name in (stages if args.command == "all" else [args.command]):
        t0 = time.time()
        stages[name](cfg)
        print(f"[{name}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
