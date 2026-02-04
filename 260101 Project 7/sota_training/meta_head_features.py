"""
Meta-head feature engineering for Alpha-2 CatBoost routing.

Schema: schema_version=1, Group 1 (global), Group 2 (per-expert TopM), Group 3 (global).
Output: list of dicts (one per sample) or DataFrame; feature_names + schema_version saved next to .cbm.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_TOP_M = 4


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def _p1_p2_margin_entropy_logitgap(
    probs: np.ndarray, logits: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """One expert, one sample. probs/logits shape (C,)"""
    p1 = float(np.max(probs))
    idx = np.argsort(probs.ravel())[::-1]
    p2 = float(probs.ravel()[idx[1]]) if len(idx) > 1 else 0.0
    margin = p1 - p2
    entropy = _entropy_from_probs(probs.ravel())
    logit_max = float(np.max(logits))
    logit_2nd = float(np.sort(logits.ravel())[-2]) if logits.size > 1 else logit_max
    logit_gap = logit_max - logit_2nd
    return p1, p2, margin, entropy, logit_max, logit_gap


def compute_y_route_one_sample(
    all_probs: List[Union[torch.Tensor, np.ndarray]],
    all_logits: List[Union[torch.Tensor, np.ndarray]],
    label_subsets: List[List[int]],
    y_true: int,
    num_active: int,
) -> int:
    """
    y_route: if any expert correct -> argmax p1 among correct; else argmax margin.
    Tie-break: smaller expert index.
    """
    E = min(num_active, len(all_probs), len(label_subsets))
    p1_list = []
    margin_list = []
    pred_global_list = []
    for e in range(E):
        probs_e = _to_numpy(all_probs[e]).reshape(-1)
        logits_e = _to_numpy(all_logits[e]).reshape(-1)
        subset = label_subsets[e]
        pred_local = int(np.argmax(logits_e))
        pred_global = subset[pred_local]
        pred_global_list.append(pred_global)
        p1_list.append(float(np.max(probs_e)))
        idx = np.argsort(probs_e)[::-1]
        p2 = float(probs_e[idx[1]]) if len(idx) > 1 else 0.0
        margin_list.append(p1_list[-1] - p2)

    correct_mask = [pred_global_list[e] == y_true for e in range(E)]
    if any(correct_mask):
        best_e = 0
        best_p1 = -1.0
        for e in range(E):
            if correct_mask[e] and p1_list[e] > best_p1:
                best_p1 = p1_list[e]
                best_e = e
        return best_e
    else:
        best_e = int(np.argmax(margin_list))
        return best_e


def build_meta_features_one_sample(
    all_logits: List[Union[torch.Tensor, np.ndarray]],
    all_probs: List[Union[torch.Tensor, np.ndarray]],
    label_subsets: List[List[int]],
    num_active_experts: int,
    phase_id: int,
    reserved_expert_indices: Optional[Set[int]] = None,
    top_m: int = DEFAULT_TOP_M,
) -> Dict[str, Any]:
    """
    Build one row of meta features for a single sample.
    all_logits / all_probs: list of length E, each shape (1, C_e) or (C_e,).
    """
    reserved = reserved_expert_indices or set()
    E = min(num_active_experts, len(all_logits), len(all_probs), len(label_subsets))
    M = min(top_m, E)

    per_e = []
    for e in range(E):
        probs_e = _to_numpy(all_probs[e]).reshape(-1)
        logits_e = _to_numpy(all_logits[e]).reshape(-1)
        p1, p2, margin, entropy, logit_max, logit_gap = _p1_p2_margin_entropy_logitgap(
            probs_e, logits_e
        )
        per_e.append(
            {
                "p1": p1,
                "p2": p2,
                "margin": margin,
                "entropy": entropy,
                "logit_max": logit_max,
                "logit_gap": logit_gap,
            }
        )

    p1_values = [per_e[e]["p1"] for e in range(E)]
    margin_values = [per_e[e]["margin"] for e in range(E)]
    order_by_p1 = np.argsort(p1_values)[::-1]
    top_m_indices = order_by_p1[:M].tolist()

    best_expert_by_p1 = int(np.argmax(p1_values))
    best_expert_by_margin = int(np.argmax(margin_values))
    sorted_margin_order = np.argsort(margin_values)[::-1]
    rank_best_expert_by_margin = int(
        np.where(sorted_margin_order == best_expert_by_p1)[0][0]
    )

    best_p1 = per_e[best_expert_by_p1]["p1"]
    second_p1 = (
        sorted(p1_values, reverse=True)[1]
        if len(p1_values) > 1
        else best_p1
    )
    gap_best_second = best_p1 - second_p1
    best_margin = per_e[best_expert_by_margin]["margin"]
    second_margin = (
        sorted(margin_values, reverse=True)[1]
        if len(margin_values) > 1
        else best_margin
    )
    best_entropy = per_e[best_expert_by_p1]["entropy"]

    reserved_expert_flag = 1 if best_expert_by_p1 in reserved else 0
    row = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": phase_id,
        "num_active_experts": E,
        "reserved_expert_flag": reserved_expert_flag,
    }
    for i, e in enumerate(top_m_indices):
        prefix = f"e{i}_"
        row[prefix + "p1"] = per_e[e]["p1"]
        row[prefix + "p2"] = per_e[e]["p2"]
        row[prefix + "margin"] = per_e[e]["margin"]
        row[prefix + "entropy"] = per_e[e]["entropy"]
        row[prefix + "logit_max"] = per_e[e]["logit_max"]
        row[prefix + "logit_gap"] = per_e[e]["logit_gap"]
    row["best_p1"] = best_p1
    row["second_p1"] = second_p1
    row["gap_best_second"] = gap_best_second
    row["best_margin"] = best_margin
    row["second_margin"] = second_margin
    row["best_entropy"] = best_entropy
    row["best_expert_id_by_p1"] = best_expert_by_p1
    row["best_expert_id_by_margin"] = best_expert_by_margin
    row["rank_best_expert_by_margin"] = rank_best_expert_by_margin

    return row


def build_meta_features(
    signals_dict: Dict[str, Any],
    phase_id: int,
    num_active_experts: Optional[int] = None,
    reserved_expert_indices: Optional[Set[int]] = None,
    top_m: int = DEFAULT_TOP_M,
) -> List[Dict[str, Any]]:
    """
    Build meta features for a batch. signals_dict from model.forward_expert_signals(video).
    Returns list of dicts (one per sample); can be converted to DataFrame with fixed column order.
    """
    all_logits = signals_dict["all_logits"]
    all_probs = signals_dict["all_probs"]
    label_subsets = signals_dict["label_subsets"]
    E = num_active_experts or signals_dict.get("num_active_experts", len(all_logits))
    B = all_logits[0].shape[0] if hasattr(all_logits[0], "shape") else 1

    rows = []
    for b in range(B):
        all_logits_b = [
            lg[b : b + 1] if hasattr(lg, "shape") and lg.ndim > 1 else lg
            for lg in all_logits
        ]
        all_probs_b = [
            pr[b : b + 1] if hasattr(pr, "shape") and pr.ndim > 1 else pr
            for pr in all_probs
        ]
        row = build_meta_features_one_sample(
            all_logits_b,
            all_probs_b,
            label_subsets,
            num_active_experts=E,
            phase_id=phase_id,
            reserved_expert_indices=reserved_expert_indices,
            top_m=top_m,
        )
        rows.append(row)
    return rows


def get_feature_column_order(top_m: int = DEFAULT_TOP_M) -> List[str]:
    """Fixed column order for schema_version=1 (for DataFrame and CatBoost)."""
    cols = [
        "schema_version",
        "phase_id",
        "num_active_experts",
        "reserved_expert_flag",
    ]
    for i in range(top_m):
        cols.extend(
            [
                f"e{i}_p1",
                f"e{i}_p2",
                f"e{i}_margin",
                f"e{i}_entropy",
                f"e{i}_logit_max",
                f"e{i}_logit_gap",
            ]
        )
    cols.extend(
        [
            "best_p1",
            "second_p1",
            "gap_best_second",
            "best_margin",
            "second_margin",
            "best_entropy",
            "best_expert_id_by_p1",
            "best_expert_id_by_margin",
            "rank_best_expert_by_margin",
        ]
    )
    return cols


def extract_expert_signals(
    model: torch.nn.Module,
    video_batch: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Call model.forward_expert_signals(video_batch) and return dict.
    Model must be VideoMAEAlphaExpertsForAction (or compatible).
    """
    model.eval()
    with torch.no_grad():
        video_batch = video_batch.to(device, non_blocking=True)
        if hasattr(model, "forward_expert_signals"):
            return model.forward_expert_signals(video_batch)
        raise AttributeError(
            "Model has no forward_expert_signals; use VideoMAEAlphaExpertsForAction."
        )
