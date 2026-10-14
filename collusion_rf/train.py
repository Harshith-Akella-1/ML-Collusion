"""Train the three detectors, tune with GroupKFold, select by validation PR-AUC, pick the F0.5 threshold."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RunConfig
from .dataset import load, subset
from .features import FEATURE_COLS

MODEL_NAMES = ["logreg", "rf", "hgb"]


def xy(feats) -> tuple[np.ndarray, np.ndarray]:
    """Model inputs: exactly the 15 FEATURE_COLS, and the label."""
    return feats[FEATURE_COLS].to_numpy(dtype=float), feats["label"].to_numpy(dtype=int)


def best_threshold(y: np.ndarray, prob: np.ndarray, beta: float = 0.5) -> tuple[float, float]:
    """Threshold maximising F-beta (beta=0.5 weights precision: false accusations are costly)."""
    prec, rec, thr = precision_recall_curve(y, prob)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta ** 2) * prec * rec / np.maximum(beta ** 2 * prec + rec, 1e-12)
    return float(thr[np.argmax(f)]), float(f.max())


def build_models(cfg: RunConfig) -> dict:
    """The three estimators of SPEC section 8 (rf and hgb wrapped in a grouped grid search)."""
    def grid(est, params):
        return GridSearchCV(est, params, cv=GroupKFold(n_splits=5), scoring="average_precision")
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "rf": grid(RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=cfg.seed),
                   {"n_estimators": [cfg.rf_trees], "max_depth": [8, 16, None], "min_samples_leaf": [1, 5, 20]}),
        "hgb": grid(HistGradientBoostingClassifier(class_weight="balanced", random_state=cfg.seed),
                    {"learning_rate": [0.05, 0.1], "max_depth": [3, 6]}),
    }


def train(cfg: RunConfig) -> None:
    """Fit, select and save models plus the decision threshold."""
    feats, _, splits = load(cfg)
    tr, va = subset(feats, splits["train"]), subset(feats, splits["val"])
    (Xtr, ytr), (Xva, yva) = xy(tr), xy(va)
    out = Path(cfg.out) / "models"
    out.mkdir(parents=True, exist_ok=True)
    fitted, val_ap, val_prob = {}, {}, {}
    for name, est in build_models(cfg).items():
        if isinstance(est, GridSearchCV):
            est.fit(Xtr, ytr, groups=tr["run_id"].to_numpy())
            extra, est = f"   best params {est.best_params_}", est.best_estimator_
        else:
            extra = ""
            est.fit(Xtr, ytr)
        fitted[name], val_prob[name] = est, est.predict_proba(Xva)[:, 1]
        val_ap[name] = float(average_precision_score(yva, val_prob[name]))
        joblib.dump(est, out / f"{name}.joblib", compress=3)
        print(f"[train] {name:<7} val PR-AUC {val_ap[name]:.3f}{extra}")
    best = max(val_ap, key=val_ap.get)
    tau, f05 = best_threshold(yva, val_prob[best])
    joblib.dump(fitted[best], out / "best_model.joblib", compress=3)
    (out / "threshold.json").write_text(json.dumps({"model": best, "tau": tau, "val_pr_auc": val_ap}))
    print(f"[train] selected: {best}   tau (F0.5-optimal) = {tau:.3f}   val F0.5 = {f05:.3f}")


def load_models(cfg: RunConfig) -> tuple[dict, str, float]:
    """Return ({name: model}, best model name, tau)."""
    d = Path(cfg.out) / "models"
    th = json.loads((d / "threshold.json").read_text())
    return {n: joblib.load(d / f"{n}.joblib") for n in MODEL_NAMES}, th["model"], th["tau"]
