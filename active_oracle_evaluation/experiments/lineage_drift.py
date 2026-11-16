"""Measure how much of the V2 warm start survives in the AlphaZero and PPO checkpoints.

For each tensor, the experiment reports two numbers against a reference checkpoint:

    relative drift   ||w - w0|| / ||w0||     how far the weights moved
    cosine           <w, w0> / (|w| |w0|)    whether they still point the same way

To give the cosine a scale, it also compares a freshly initialized network of the same
architecture against the same reference. Finally, it reports how often each agent picks the
same move as ``v2`` on the decision suite, read from the cached choices in the score cache.

``OutcomeMLP`` names its layers ``body.*`` and ``PolicyValueMLP`` names them ``trunk.*``. They are
the same three hidden layers, so the comparison maps them explicitly (``body.6`` is the outcome
head, which maps to ``outcome_head``).

The settings are in the [lineage_drift] table of config/experiments.toml. Run from the repository
root:

    python -m active_oracle_evaluation.experiments.lineage_drift

The tables go to stdout and to outputs/lineage_drift/lineage_drift.md and lineage_drift.json.
"""

from __future__ import annotations

import argparse

import torch

from active_oracle_evaluation.config import settings
from active_oracle_evaluation.paths import DATA, MODELS
from active_oracle_evaluation.results import write_results

EXPERIMENT = "lineage_drift"
SCORE_CACHE = DATA / "decision_suites" / "full_suite_scores_0ply_large.npz"

# OutcomeMLP tensor -> PolicyValueMLP tensor. The three hidden layers are the shared trunk;
# body.6 is v2's outcome head and lands on the policy-value net's outcome head.
OUTCOME_TO_PV = {
    "body.0.weight": "trunk.0.weight",
    "body.0.bias": "trunk.0.bias",
    "body.2.weight": "trunk.2.weight",
    "body.2.bias": "trunk.2.bias",
    "body.4.weight": "trunk.4.weight",
    "body.4.bias": "trunk.4.bias",
    "body.6.weight": "outcome_head.weight",
    "body.6.bias": "outcome_head.bias",
}

TRUNK = ("trunk.0", "trunk.2", "trunk.4")
BASELINE_LABEL = "v2 -> random init (baseline)"
TENSOR_HEADER = ["tensor", "drift", "cosine"]


def _state(name: str) -> dict:
    """Load the state dict of checkpoint ``models/<name>.pt``."""
    return torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=True)["state_dict"]


def _meta(name: str) -> dict:
    """Load the hyperparameters and game count recorded in checkpoint ``models/<name>.pt``."""
    p = torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=True)
    return {"hyperparams": p.get("hyperparams", {}), "games_trained": p.get("games_trained")}


def _compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Return (relative drift, cosine) of ``b`` against reference ``a``.

    Both are moved to CPU first, because MPS has no float64 and the norms are taken in float64.
    """
    x = a.detach().cpu().flatten().double()
    y = b.detach().cpu().flatten().double()
    return float((y - x).norm() / x.norm()), float(torch.dot(x, y) / (x.norm() * y.norm()))


def _pairs(ref_sd: dict, other_sd: dict, mapping: dict[str, str] | None):
    """Yield (label, ref_tensor, other_tensor) for comparable tensors."""
    keys = mapping or {k: k for k in ref_sd}
    for rk, ok in keys.items():
        if rk not in ref_sd or ok not in other_sd:
            continue
        if ref_sd[rk].shape != other_sd[ok].shape:
            continue
        if ref_sd[rk].dtype not in (torch.float32, torch.float64):
            continue
        if ref_sd[rk].flatten().double().norm() == 0:
            continue  # constant buffers (class_points) carry no lineage signal
        yield ok, ref_sd[rk], other_sd[ok]


def compare_states(ref_sd: dict, other_sd: dict, mapping: dict[str, str] | None) -> dict:
    """Return {tensor: {"drift", "cosine"}} for every comparable tensor pair."""
    return {
        label: dict(zip(("drift", "cosine"), _compare(r, o)))
        for label, r, o in _pairs(ref_sd, other_sd, mapping)
    }


def random_init_baseline(ref_sd: dict, *, seed: int) -> dict:
    """Compare a freshly initialized network of the same architecture against ``ref_sd``.

    The network is built with the model class itself, so it uses that class's own initializer.
    """
    from active_oracle_evaluation.backgammon.nn.policy_value_mlp import PolicyValueMLP

    torch.manual_seed(seed)
    arch = torch.load(MODELS / "policy_value_v2init.pt", map_location="cpu", weights_only=True)[
        "arch"
    ]
    fresh = PolicyValueMLP(**arch).state_dict()
    return compare_states(ref_sd, fresh, OUTCOME_TO_PV)


def choice_agreement(cfg: dict) -> dict:
    """Return how often each agent chooses the same move as the reference agent on the suite.

    Reads each agent's chosen action per position from the score cache. ``v2init`` shares the
    ``v2`` trunk exactly, so it gives the ceiling of this measure.
    """
    import numpy as np

    d = np.load(SCORE_CACHE, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    idx = {n: i for i, n in enumerate(names)}
    choices = np.asarray(d["choices"], dtype=int)
    n_actions = np.asarray(d["n_actions"], dtype=int)
    real = n_actions >= cfg["min_legal_moves"]
    reference = cfg["reference_agent"]

    def agree(a: str, b: str) -> float:
        """Fraction of the counted positions where ``a`` and ``b`` choose the same move."""
        return float((choices[idx[a]][real] == choices[idx[b]][real]).mean())

    return {
        "n_positions": int(real.sum()),
        "chance": float(np.mean(1.0 / n_actions[real])),
        "lineage": {n: agree(reference, n) for n in cfg["lineage_agents"] if n in idx},
        "outside": {n: agree(reference, n) for n in cfg["outside_agents"] if n in idx},
    }


def weight_results(cfg: dict) -> dict:
    """Run every configured comparison, then the fresh-init baseline, in that order."""
    states: dict[str, dict] = {}
    results: dict[str, dict] = {}
    for comparison in cfg["comparisons"]:
        reference = comparison["reference"]
        if reference not in states:
            states[reference] = _state(reference)
        mapping = OUTCOME_TO_PV if comparison["outcome_to_policy_value"] else None
        target_sd = _state(comparison["target"])
        results[comparison["label"]] = compare_states(states[reference], target_sd, mapping)
    reference_sd = _state(cfg["reference_agent"])
    results[BASELINE_LABEL] = random_init_baseline(reference_sd, seed=cfg["baseline_seed"])
    return results


def print_weight_tables(results: dict, tensors: list[str]) -> list:
    """Print one drift/cosine table per comparison and return them as Markdown tables."""
    print("Weight drift across the V2 -> AlphaZero / PPO lineage")
    print("relative drift = ||w-w0||/||w0||   cosine = directional agreement\n")
    width = max(len(t) for t in tensors) + 2
    tables = []
    for label, table in results.items():
        print(label)
        rows = []
        for t in tensors:
            if t not in table:
                continue
            drift, cosine = table[t]["drift"], table[t]["cosine"]
            print(f"  {t:<{width}}drift {drift:>7.3f}   cosine {cosine:>8.4f}")
            rows.append([t, f"{drift:.3f}", f"{cosine:.4f}"])
        print()
        tables.append((label, TENSOR_HEADER, rows))
    return tables


def _trunk_cosines(table: dict) -> list[float]:
    """Return the cosines of the trunk tensors in one comparison table."""
    return [v["cosine"] for k, v in table.items() if k.startswith(TRUNK)]


def print_summary(results: dict) -> tuple:
    """Print the trunk and head cosine summary and return it as a Markdown table."""
    trunk_cos = _trunk_cosines(results["v2 -> az75"])
    base_cos = _trunk_cosines(results[BASELINE_LABEL])
    az75_heads = results["v2init -> az75"]
    policy_cos = az75_heads["policy_head.weight"]["cosine"]
    outcome_cos = az75_heads["outcome_head.weight"]["cosine"]
    ppo_cos = min(_trunk_cosines(results["v2 -> ppo"]))
    rows = [
        ["az75 trunk vs v2", f"cosine {min(trunk_cos):.4f}-{max(trunk_cos):.4f}"],
        ["random init vs v2 (trunk)", f"cosine {min(base_cos):+.4f}-{max(base_cos):+.4f}"],
        ["az75 heads vs v2init", f"policy {policy_cos:.4f}, outcome {outcome_cos:.4f}"],
        ["ppo trunk vs v2", f"cosine {ppo_cos:.4f}"],
    ]
    print("Summary")
    print(f"  az75 trunk vs v2:          {rows[0][1]}")
    print(f"  random init vs v2 (trunk): {rows[1][1]}")
    print(f"  az75 heads vs v2init:      {rows[2][1]}")
    print(f"  ppo trunk vs v2:           {rows[3][1]}")
    return ("Summary", ["comparison", "cosine"], rows)


def print_provenance(checkpoints: list[str]) -> tuple:
    """Print the training metadata recorded in each checkpoint and return it as a table."""
    print("\nProvenance recorded in checkpoint metadata")
    rows = []
    for n in checkpoints:
        m = _meta(n)
        hp = m["hyperparams"]
        init = {k: v for k, v in hp.items() if "init" in k or k == "freeze_value"}
        print(f"  {n:<22} games={m['games_trained']:<7} init-fields={init or '(none recorded)'}")
        rows.append([n, str(m["games_trained"]), str(init or "(none recorded)")])
    return ("Provenance", ["checkpoint", "games", "init fields"], rows)


def print_agreement(ca: dict) -> list:
    """Print the move agreement with the reference agent and return it as Markdown tables."""
    print(
        f"\nChoice agreement with v2 on {ca['n_positions']} suite positions "
        f"(chance {ca['chance']:.1%})"
    )
    rows = []
    for n, v in ca["lineage"].items():
        print(f"  {n:<12}{v:>7.1%}   in the v2 lineage")
        rows.append([n, f"{v:.1%}", "yes"])
    for n, v in ca["outside"].items():
        print(f"  {n:<12}{v:>7.1%}")
        rows.append([n, f"{v:.1%}", "no"])
    suite = [[str(ca["n_positions"]), f"{ca['chance']:.1%}"]]
    return [
        ("Choice agreement suite", ["positions", "chance"], suite),
        ("Choice agreement with v2", ["agent", "agreement", "in the v2 lineage"], rows),
    ]


def main() -> None:
    """Compare the checkpoints, print the tables and write them as Markdown and JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    cfg = settings(EXPERIMENT)

    results = weight_results(cfg)
    tensors = list(results[cfg["comparisons"][0]["label"]])
    tables = print_weight_tables(results, tensors)
    tables.append(print_summary(results))
    tables.append(print_provenance(cfg["provenance_checkpoints"]))

    ca = choice_agreement(cfg)
    results["choice_agreement"] = ca
    tables += print_agreement(ca)

    md_path, json_path = write_results(EXPERIMENT, tables, results)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
