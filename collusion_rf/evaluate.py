"""Metrics on ID test and OOD sets, subgroup analysis and figures 1-6."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from .config import RunConfig
from .dataset import load, subset
from .features import FEATURE_COLS
from .simulate import History, sample_scenario, simulate_run
from .train import load_models, xy

KAPPA_BUCKETS = [(0.3, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0 + 1e-9)]   # SPEC buckets + the [0.3, 0.4) tail
ABLATED = ["markup_mean", "markup_std", "rel_price", "market_markup"]     # price-level features


def ablation(model, train, test, ood) -> dict:
    """Retrain the selected model (same hyperparameters) without the price-level features."""
    keep = [i for i, f in enumerate(FEATURE_COLS) if f not in ABLATED]
    (Xtr, ytr), (Xte, yte), (Xo, yo) = xy(train), xy(test), xy(ood)
    m = clone(model).fit(Xtr[:, keep], ytr)
    return {"dropped": ABLATED,
            "id_test_pr_auc": float(average_precision_score(yte, m.predict_proba(Xte[:, keep])[:, 1])),
            "ood_pr_auc": float(average_precision_score(yo, m.predict_proba(Xo[:, keep])[:, 1]))}


def scores(y: np.ndarray, prob: np.ndarray, tau: float | None = None) -> dict:
    """Threshold-free metrics, plus precision/recall/F/confusion matrix at tau if given."""
    out = {"roc_auc": float(roc_auc_score(y, prob)), "pr_auc": float(average_precision_score(y, prob))}
    if tau is not None:
        pred = prob >= tau
        out.update(precision=float(precision_score(y, pred, zero_division=0)),
                   recall=float(recall_score(y, pred, zero_division=0)),
                   f1=float(fbeta_score(y, pred, beta=1, zero_division=0)),
                   f05=float(fbeta_score(y, pred, beta=0.5, zero_division=0)),
                   confusion=confusion_matrix(y, pred, labels=[0, 1]).tolist())
    return out


def subgroups(df, pred: np.ndarray) -> dict:
    """Recall by kappa bucket and false-positive rates for the hard negatives (rate = share flagged)."""
    y, k, member = df["label"].to_numpy() == 1, df["kappa"].to_numpy(), df["in_cartel"].to_numpy()
    active = df["frac_active"].to_numpy() >= 0.5
    groups = {f"recall_kappa_{lo:.1f}_{hi:.1f}": y & (k >= lo) & (k < hi) for lo, hi in KAPPA_BUCKETS}
    groups.update(fpr_umbrella=~member & (k >= 0),                  # outsiders in a cartel run (SPEC def.)
                  fpr_umbrella_active=~member & (k >= 0) & active,  # ... while the cartel is actually active
                  fpr_competitive_runs=k < 0,
                  fpr_members_inactive=member & ~y)
    return {name: {"rate": float(pred[m].mean()) if m.any() else None, "n": int(m.sum())}
            for name, m in groups.items()}


def plot_prices(ax, h: History, title: str, flags: list = ()) -> None:
    """Price paths of one run; shades active-cartel (light) and punishment (dark) periods, marks flags."""
    t, state = np.arange(len(h.cost_true)), np.array(h.state)
    for i in range(h.n):
        ax.plot(t, h.price[:, i], lw=0.8, ls="-" if h.in_cartel[i] else "--",
                label=f"firm {i}" + (" (cartel)" if h.in_cartel[i] else ""))
    for mask, color, lab in ((h.regime, "tab:red", "cartel active"), (state == "PUNISH", "black", "punishment")):
        ax.fill_between(t, 0, 1, where=mask, transform=ax.get_xaxis_transform(), color=color, alpha=0.12, label=lab)
    for f in flags:
        ax.axvline(f["t"], color="tab:green" if f["guilty"] else "tab:orange", lw=1.2, ls=":")
    ax.set(title=title, xlabel="t", ylabel="price")
    ax.legend(fontsize=6, ncol=3, loc="upper left")


def _save(fig, cfg: RunConfig, name: str) -> None:
    d = Path(cfg.out) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(d / name, dpi=130)
    plt.close(fig)


def _curve_figs(cfg, y_id, y_ood, probs, best) -> None:
    for fname, curve, xl, yl in (("pr_curves.png", precision_recall_curve, "recall", "precision"),
                                 ("roc_curves.png", roc_curve, "false positive rate", "true positive rate")):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for name, (p_id, p_ood) in probs.items():
            series = [(y_id, p_id, f"{name} (ID test)", "-")] + ([(y_ood, p_ood, f"{name} (OOD)", "--")] * (name == best))
            for y, p, lab, ls in series:
                a, b, _ = curve(y, p)
                ax.plot(*((b, a) if fname.startswith("pr") else (a, b)), ls, label=lab)
        ax.set(xlabel=xl, ylabel=yl, title=fname[:-4].replace("_", " "))
        ax.legend(fontsize=8)
        _save(fig, cfg, fname)


def _example_runs_fig(cfg: RunConfig) -> None:
    comp = simulate_run(40000, sample_scenario(40000, cfg.seed, cartel=False), cfg.horizon, cfg.seed).history
    cands = (simulate_run(r, sample_scenario(r, cfg.seed, cartel=True), cfg.horizon, cfg.seed).history
             for r in range(40001, 40040))
    cart = next(h for i, h in enumerate(cands) if ("PUNISH" in h.state and h.regime.mean() > 0.3) or i == 38)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    plot_prices(axes[0], comp, "Competitive run")
    plot_prices(axes[1], cart, "Cartel run (solid = cartel members, dashed = outsiders)")
    _save(fig, cfg, "example_runs.png")


def evaluate(cfg: RunConfig) -> dict:
    """Write metrics.json and figures 1-6; print the summary table."""
    feats, ood, splits = load(cfg)
    test = subset(feats, splits["test"])
    models, best, tau = load_models(cfg)
    (X_id, y_id), (X_ood, y_ood) = xy(test), xy(ood)
    probs = {n: (m.predict_proba(X_id)[:, 1], m.predict_proba(X_ood)[:, 1]) for n, m in models.items()}
    metrics = {"best_model": best, "tau": tau, "models": {
        n: {"id_test": scores(y_id, p_id, tau if n == best else None),
            "ood": scores(y_ood, p_ood, tau if n == best else None)} for n, (p_id, p_ood) in probs.items()}}
    pred = probs[best][0] >= tau
    sub = metrics["subgroups"] = subgroups(test, pred)
    imp = permutation_importance(models[best], X_id, y_id, scoring="average_precision", n_repeats=5,
                                 random_state=cfg.seed)
    metrics["permutation_importance"] = dict(zip(FEATURE_COLS, imp.importances_mean.tolist()))
    abl = metrics["ablation"] = ablation(models[best], subset(feats, splits["train"]), test, ood)
    rep = Path(cfg.out) / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _curve_figs(cfg, y_id, y_ood, probs, best)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cm = np.array(metrics["models"][best]["id_test"]["confusion"])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["competitive", "colluding"],
           yticklabels=["competitive", "colluding"], xlabel="predicted", ylabel="true",
           title=f"{best}, ID test, tau={tau:.2f}")
    _save(fig, cfg, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    order = np.argsort(imp.importances_mean)
    ax.barh(np.array(FEATURE_COLS)[order], imp.importances_mean[order], xerr=imp.importances_std[order])
    ax.set(xlabel="drop in PR-AUC when permuted", title=f"Permutation importance ({best}, ID test)")
    _save(fig, cfg, "feature_importance.png")
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    keys = [k for k in sub if k.startswith("recall_kappa")]
    ax.bar([k[13:].replace("_", "–") for k in keys], [sub[k]["rate"] or 0 for k in keys])
    ax.set(xlabel="collusion strength kappa", ylabel="recall", ylim=(0, 1), title=f"Recall by kappa ({best})")
    _save(fig, cfg, "recall_by_kappa.png")
    _example_runs_fig(cfg)

    for n in models:
        a, b = metrics["models"][n]["id_test"], metrics["models"][n]["ood"]
        print(f"[evaluate] {n:<7} ID test ROC-AUC {a['roc_auc']:.3f} PR-AUC {a['pr_auc']:.3f} | "
              f"OOD ROC-AUC {b['roc_auc']:.3f} PR-AUC {b['pr_auc']:.3f}")
    for split in ("id_test", "ood"):
        m = metrics["models"][best][split]
        print(f"[evaluate] {best} @ tau={tau:.3f} {split:<8}: precision {m['precision']:.3f} recall "
              f"{m['recall']:.3f} F1 {m['f1']:.3f} F0.5 {m['f05']:.3f} confusion {m['confusion']}")
    print("[evaluate] " + "  ".join(f"{k}={v['rate']:.3f} (n={v['n']})" if v["rate"] is not None
                                    else f"{k}=n/a" for k, v in sub.items()))
    print(f"[evaluate] ablation: {best} retrained without {', '.join(ABLATED)}: "
          f"ID test PR-AUC {abl['id_test_pr_auc']:.3f} | OOD PR-AUC {abl['ood_pr_auc']:.3f}")
    print(f"[evaluate] figures written to {Path(cfg.out) / 'figures'}")
    return metrics
