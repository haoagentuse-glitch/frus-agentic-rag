"""Pick the smallest gold subset that still reaches the full set's conclusion.

The ablation costs about two hours per variant, and most of that buys nothing:
a case that every configuration answers, or that every configuration fails,
contributes the same number to every mean and cannot separate anything. Only
cases the configurations *disagree* on carry information about which
configuration is better.

So this reads the runs already recorded (reports/runs_*.jsonl), builds a
case x configuration matrix, and searches for a subset that reproduces the
full-set numbers. Two things it deliberately does, because a smaller set that
answers a different question is worse than no shortcut at all:

- Zero-variance cases are dropped first. That is the dimensionality reduction;
  everything after it is refinement.
- Selection is greedy on the metric that matters — the per-configuration mean —
  and the script reports the residual error and whether the configuration
  ranking survived. A subset is only usable if it did.

Cases needed to evaluate a gate are pinned regardless of variance: gate 3 is
about unanswerable questions, and a subset with none cannot report it. The same
holds for keeping both languages and every question kind represented, since the
findings are reported per slice.

    python scripts/select_core_cases.py [--size 12]

Writes eval/gold_cases_core.jsonl, runnable directly:

    FRUS_GPU=1 ./scripts/dev.sh frus eval --cases eval/gold_cases_core.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPORTS = Path("reports")
CASES = Path("eval/gold_cases.jsonl")
OUT = Path("eval/gold_cases_core.jsonl")


def load_matrix() -> tuple[dict, dict, list[str]]:
    """M[(case_id, language)][config] = correctness in 0..1.

    A configuration is (variant, system): the subset has to preserve differences
    along both axes, since one is what the sweep compares and the other is what
    the ablation compares.
    """
    matrix: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    meta: dict[tuple[str, str], dict] = {}
    configs: set[str] = set()
    for path in sorted(REPORTS.glob("runs_*.jsonl")):
        variant = path.stem.replace("runs_", "")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            # A judge that errored measured nothing; treating it as 0 is the bug
            # that made one whole variant look like a regression.
            if r.get("judge") == "error" or r.get("judge_score") is None:
                continue
            key = (r["case_id"], r["language"])
            cfg = f"{variant}/{r['system']}"
            matrix[key][cfg] = float(r["judge_score"])
            configs.add(cfg)
            meta.setdefault(
                key,
                {
                    "kind": r["kind"],
                    "language": r["language"],
                    "answerable": r["answerable"],
                },
            )
    return matrix, meta, sorted(configs)


def config_means(keys: list, matrix: dict, configs: list[str]) -> dict[str, float]:
    out = {}
    for c in configs:
        vals = [matrix[k][c] for k in keys if c in matrix[k]]
        out[c] = sum(vals) / len(vals) if vals else 0.0
    return out


def max_error(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(a[c] - b[c]) for c in a)


def rank_agreement(a: dict[str, float], b: dict[str, float], decisive_pp: float = 0.0):
    """Fraction of configuration pairs ordered the same way in both.

    Ordering is the property that has to survive, since the subset exists to
    pick a winner. `decisive_pp` restricts the count to pairs the full set
    actually separates: these configurations sit within nine points of each
    other, so scoring a subset on whether it reproduces a 0.3pp gap measures
    noise in both, not fidelity. Report both — the unfiltered number says how
    jittery the ranking is, the decisive one says whether a real difference
    would survive.
    """
    cs = sorted(a)
    same = total = 0
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            x, y = cs[i], cs[j]
            full, sub = a[x] - a[y], b[x] - b[y]
            if abs(full) * 100 < decisive_pp:
                continue
            total += 1
            if (full > 0) == (sub > 0):
                same += 1
    return (same / total if total else 1.0), total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=12, help="target number of cases (not case-langs)")
    args = ap.parse_args()

    matrix, meta, configs = load_matrix()
    if not configs:
        raise SystemExit("no runs_*.jsonl with judged scores; run the ablation first")
    keys = sorted(matrix)
    full = config_means(keys, matrix, configs)

    # Step 1: the reduction. A case-language whose score is identical under every
    # configuration shifts all means equally and can never change a comparison.
    def spread(k) -> float:
        v = list(matrix[k].values())
        return max(v) - min(v) if v else 0.0

    informative = [k for k in keys if spread(k) > 0]
    dead = len(keys) - len(informative)

    # Step 2: pin what the gates and the per-slice reporting need, whatever their
    # variance. Gate 3 is defined on unanswerable questions; a subset without any
    # cannot report it at all.
    pinned: list = []
    for kind in sorted({m["kind"] for m in meta.values()}):
        for lang in ("zh-TW", "en"):
            cands = [k for k in keys if meta[k]["kind"] == kind and meta[k]["language"] == lang]
            if cands:
                pinned.append(max(cands, key=spread))
    pinned = list(dict.fromkeys(pinned))

    # Step 3: greedily add the case-language that most reduces the error between
    # the subset's per-configuration means and the full set's.
    selected = list(pinned)
    pool = [k for k in informative if k not in selected]
    target_keys = args.size * 2  # both languages per case
    while len(selected) < target_keys and pool:
        best, best_err = None, None
        for k in pool:
            err = max_error(full, config_means([*selected, k], matrix, configs))
            if best_err is None or err < best_err:
                best, best_err = k, err
        selected.append(best)
        pool.remove(best)

    sub = config_means(selected, matrix, configs)
    case_ids = sorted({c for c, _ in selected})

    # The fidelity curve. A subset is a trade, and the only honest way to offer
    # it is to show where it stops being safe rather than to assert a size.
    curve = []
    trial = list(pinned)
    tpool = [k for k in informative if k not in trial]
    while tpool:
        best, best_err = None, None
        for k in tpool:
            err = max_error(full, config_means([*trial, k], matrix, configs))
            if best_err is None or err < best_err:
                best, best_err = k, err
        trial.append(best)
        tpool.remove(best)
        m = config_means(trial, matrix, configs)
        loose, _ = rank_agreement(full, m)
        strict, npairs = rank_agreement(full, m, decisive_pp=3.0)
        curve.append(
            {
                "case_langs": len(trial),
                "speedup": round(len(keys) / len(trial), 2),
                "max_err_pp": round(max_error(full, m) * 100, 2),
                "rank_all": round(loose, 3),
                "rank_decisive_3pp": round(strict, 3),
                "decisive_pairs": npairs,
            }
        )

    rows = [json.loads(x) for x in CASES.read_text().splitlines() if x.strip()]
    keep = [r for r in rows if r["case_id"] in case_ids]
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep))

    report = {
        "full_case_langs": len(keys),
        "zero_variance_dropped": dead,
        "informative": len(informative),
        "selected_case_langs": len(selected),
        "selected_cases": len(case_ids),
        "configs_compared": len(configs),
        "max_abs_error_pp": round(max_error(full, sub) * 100, 2),
        "rank_agreement_all": round(rank_agreement(full, sub)[0], 4),
        "rank_agreement_decisive_3pp": round(rank_agreement(full, sub, 3.0)[0], 4),
        "fidelity_curve": curve,
        "speedup": round(len(keys) / len(selected), 2),
        "kinds": {
            k: sum(1 for s in selected if meta[s]["kind"] == k)
            for k in sorted({m["kind"] for m in meta.values()})
        },
        "languages": {
            lang: sum(1 for s in selected if meta[s]["language"] == lang)
            for lang in ("zh-TW", "en")
        },
        "unanswerable": sum(1 for s in selected if not meta[s]["answerable"]),
    }
    (REPORTS / "core_cases.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "fidelity_curve"}, ensure_ascii=False, indent=2
        )
    )
    print("\nfidelity curve (how much the subset can be trusted at each size):")
    print(f"  {'cases':>6s} {'speedup':>8s} {'max_err':>8s} {'rank_all':>9s} {'rank>3pp':>9s}")
    for row in curve:
        if row["case_langs"] % 4 == 0 or row["case_langs"] == len(informative):
            print(
                f"  {row['case_langs'] // 2:6d} {row['speedup']:8.2f}x {row['max_err_pp']:7.2f}pp"
                f" {row['rank_all']:9.3f} {row['rank_decisive_3pp']:9.3f}"
            )
    print(f"\nwrote {len(keep)} cases to {OUT}")
    print("\nper-configuration mean, full vs subset:")
    for c in sorted(configs, key=lambda c: -full[c]):
        print(
            f"  {c:34s} full={full[c]:.3f}  subset={sub[c]:.3f}"
            f"  diff={(sub[c] - full[c]) * 100:+6.2f}pp"
        )


main()
