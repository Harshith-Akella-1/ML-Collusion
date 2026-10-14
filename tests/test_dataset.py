import json

import pytest

from collusion_rf.config import RunConfig
from collusion_rf.dataset import generate, load, subset
from collusion_rf.features import FEATURE_COLS


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    cfg = RunConfig(n_id=40, n_ood=8, horizon=200, out=str(tmp_path_factory.mktemp("out")))
    generate(cfg)
    return cfg, *load(cfg)


def test_no_run_overlap_between_splits(data):
    cfg, feats_id, feats_ood, splits = data
    names = ["train", "val", "test", "ood"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(splits[a]) & set(splits[b]), (a, b)
    assert set(splits["train"]) | set(splits["val"]) | set(splits["test"]) == set(feats_id["run_id"])
    assert set(splits["ood"]) == set(feats_ood["run_id"])
    assert not set(feats_id["run_id"]) & set(feats_ood["run_id"])
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (28, 6, 6)


def test_rows_of_a_run_stay_in_one_split(data):
    _, feats_id, _, splits = data
    parts = [subset(feats_id, splits[k]) for k in ("train", "val", "test")]
    assert sum(len(p) for p in parts) == len(feats_id)
    assert not set(parts[0]["run_id"]) & set(parts[2]["run_id"])


def test_feature_cols_exclude_metadata_and_label():
    assert not {"kappa", "in_cartel", "frac_active", "label", "run_id"} & set(FEATURE_COLS)
    assert len(FEATURE_COLS) == 15


def test_generation_is_reproducible(data, tmp_path):
    cfg, feats_id, _, splits = data
    cfg2 = RunConfig(n_id=40, n_ood=8, horizon=200, out=str(tmp_path))
    generate(cfg2)
    feats2, _, splits2 = load(cfg2)
    assert feats_id.equals(feats2) and json.dumps(splits) == json.dumps(splits2)
