"""Generate many runs -> feature tables (ID and OOD) and run-level train/val/test splits."""
import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from .config import OOD_ID0, RunConfig
from .features import build_features
from .market import table
from .simulate import sample_scenario, simulate_run


def run_features(run_id: int, seed: int, ood: bool, horizon: int) -> pd.DataFrame:
    """Simulate one run and return its feature table."""
    sc = sample_scenario(run_id, seed, ood)
    df = simulate_run(run_id, sc, horizon, seed).df
    return build_features(df, sc["kappa"] if sc["cartel"] else -1.0)


def make_features(run_ids, seed: int, ood: bool, horizon: int) -> pd.DataFrame:
    """Feature table for many runs, simulated in parallel."""
    parts = Parallel(n_jobs=-1)(delayed(run_features)(r, seed, ood, horizon) for r in run_ids)
    return pd.concat(parts, ignore_index=True)


def make_splits(feats: pd.DataFrame, seed: int) -> dict:
    """70/15/15 split of run_ids, stratified by 'has cartel'. Rows of a run never straddle sets."""
    runs = feats.groupby("run_id")["kappa"].first()
    ids, has = runs.index.to_numpy(), (runs >= 0).to_numpy()
    train, rest, _, has_rest = train_test_split(ids, has, test_size=0.30, stratify=has, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=has_rest, random_state=seed)
    return {k: sorted(int(i) for i in v) for k, v in (("train", train), ("val", val), ("test", test))}


def generate(cfg: RunConfig) -> None:
    """Simulate ID + OOD runs, build features, write parquet files and splits.json."""
    out = Path(cfg.out) / "data"
    out.mkdir(parents=True, exist_ok=True)
    tb = table()
    print("[generate] benchmarks: " + " | ".join(
        f"n={n} p_N={tb.p_n(n, 1.0):.3f} p_M={tb.p_m(n, 1.0):.3f}" for n in (3, 4, 5)))
    feats = {}
    for name, ids, ood in (("id", range(cfg.n_id), False), ("ood", range(OOD_ID0, OOD_ID0 + cfg.n_ood), True)):
        f = feats[name] = make_features(ids, cfg.seed, ood, cfg.horizon)
        f.to_parquet(out / f"features_{name}.parquet", index=False)
        n_cartel = int((f.groupby("run_id")["kappa"].first() >= 0).sum())
        print(f"[generate] {name.upper()} runs: {f['run_id'].nunique()} (cartel runs: {n_cartel})  "
              f"rows: {len(f):,}  positive rate: {f['label'].mean():.3f}")
    splits = make_splits(feats["id"], cfg.seed)
    splits["ood"] = sorted(int(i) for i in feats["ood"]["run_id"].unique())
    sets = [set(v) for v in splits.values()]
    assert sum(len(s) for s in sets) == len(set().union(*sets)), "run_id overlap between splits"
    (out / "splits.json").write_text(json.dumps(splits))
    print(f"[generate] splits by run_id: train {len(splits['train'])} / val {len(splits['val'])} / "
          f"test {len(splits['test'])} runs  (no overlap OK)")


def load(cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load (ID features, OOD features, splits) written by `generate`."""
    d = Path(cfg.out) / "data"
    return (pd.read_parquet(d / "features_id.parquet"), pd.read_parquet(d / "features_ood.parquet"),
            json.loads((d / "splits.json").read_text()))


def subset(feats: pd.DataFrame, run_ids: list) -> pd.DataFrame:
    """Rows belonging to the given runs."""
    return feats[feats["run_id"].isin(run_ids)].reset_index(drop=True)
