"""Parts L & M -- target-user experiment, side-by-side Top-10s, and explainability.

The target user is derived from the roll number (see src/config.target_user_id).  A
reproducible 30% of that user's *training* ratings is hidden; every model is refitted on
the reduced matrix, so the target genuinely has a shorter profile while no other user's
data changes.  Ground truth for the target is the union of the hidden training ratings
and the user's global test ratings.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics as Met
from src.analysis import disagreement_items
from src.base import topk_from_scores
from src.config import CONFIG
from src.pipeline import (banner, fit_model, get_context, get_target, load_json,
                          save_json, save_table)

PARADIGMS = [("UBCF", "ubcf"), ("IBCF", "ibcf"), ("RegressionCF", "regression"),
             ("SLIM", "slim"), ("GraphRec", "graph")]
K = CONFIG.top_k


# --------------------------------------------------------------------------------------
def reason_text(label: str, expl: list[dict], titles: np.ndarray) -> str:
    if not expl:
        return "no contributing evidence in the visible profile"
    if label == "UBCF":
        parts = [f"neighbour u{e['neighbor_u']} (sim={e['similarity']:+.3f}, "
                 f"{e['co_ratings']} co-ratings) rated it {e['neighbor_rating']:.0f} "
                 f"vs their mean {e['neighbor_mean']:.2f} -> {e['contribution']:+.3f}"
                 for e in expl[:3]]
        return "strongest contributing neighbours: " + "; ".join(parts)
    if label == "IBCF":
        parts = [f"'{titles[e['neighbor_item']]}' (you rated {e['user_rating']:.0f}, "
                 f"sim={e['similarity']:+.3f}, {e['co_ratings']} co-ratings) "
                 f"-> {e['contribution']:+.3f}" for e in expl[:3]]
        return "most similar movies you already rated: " + "; ".join(parts)
    if label == "RegressionCF":
        def sig(e):
            if e.get("ranking_head") == "affinity":
                return "you liked it" if e["user_signal"] > 0 else "you did not flag it"
            return f"your deviation {e['user_signal']:+.2f}"
        parts = [f"'{titles[e['neighbor_item']]}' learned coef={e['learned_coef']:+.3f} "
                 f"(similarity {e['similarity']:+.3f}, {sig(e)}) "
                 f"-> {e['contribution']:+.3f}" for e in expl[:3]]
        return "strongest learned regression coefficients: " + "; ".join(parts)
    if label == "SLIM":
        parts = [f"'{titles[e['source_item']]}' W={e['coefficient']:.4f}"
                 for e in expl[:3]]
        return "strongest learned item-item coefficients from your liked items: " + \
            "; ".join(parts)
    if label == "GraphRec":
        parts = [f"via '{titles[e['via_item']]}' ({e['n_paths']} length-3 paths, "
                 f"weight {e['total_path_weight']:.2e})" for e in expl[:3]]
        return "strongest graph propagation paths: " + "; ".join(parts)
    return str(expl[:3])


def main():
    banner("PARTS L & M - Target-user experiment and explainability")
    ds, split, ctx = get_context()
    tgt = get_target()
    u = tgt.u
    titles = ds.titles

    # Ground truth for the target: hidden training ratings + global test ratings.
    hidden = tgt.hidden[["i", "rating"]]
    gtest = split.test[split.test.u == u][["i", "rating"]]
    truth = pd.concat([hidden, gtest], ignore_index=True)
    relevant = set(truth.loc[truth.rating >= CONFIG.relevance_threshold, "i"].astype(int))
    truth_map = dict(zip(truth.i.astype(int), truth.rating.astype(float)))

    print(f"target user id={tgt.user_id} (internal u={u})")
    print(f"  {tgt.mapping_explanation}")
    print(f"  visible profile={len(tgt.kept)} ratings, hidden={len(tgt.hidden)}, "
          f"global test={len(gtest)}, relevant held-out={len(relevant)}")

    models, recs, explanations = {}, {}, {}
    rows = []
    for label, key in PARADIGMS:
        sel = load_json(key)
        spec = sel["selection"]["ranking_selected"] if "selection" in sel \
            else sel["ranking_selected"]
        model = fit_model(spec, tgt.R_profile, split.val)
        models[label] = model

        S = model.score_all()
        topk = topk_from_scores(S, tgt.R_profile, K)[u]
        recs[label] = [int(i) for i in topk]

        P = np.clip(model.predict_all(), *ds.rating_scale)
        pred_pairs = np.array([P[u, i] for i in truth.i.astype(int)])
        rows.append({
            "model": label, "spec": str(spec),
            "rmse": Met.rmse(truth.rating.to_numpy(), pred_pairs),
            "mae": Met.mae(truth.rating.to_numpy(), pred_pairs),
            "precision_at_10": Met.precision_at_k(topk, relevant, K),
            "recall_at_10": Met.recall_at_k(topk, relevant, K),
            "hit_rate_at_10": Met.hit_rate_at_k(topk, relevant, K),
            "ndcg_at_10": Met.ndcg_at_k(topk, relevant, K),
            "map_at_10": Met.average_precision_at_k(topk, relevant, K),
            "novelty": Met.novelty(topk, ctx.item_pop, ds.n_users, K),
            "ild": Met.intra_list_diversity(topk, ctx.genre, K),
            "mean_pop_rank": Met.popularity_rank_score(topk, ctx.pop_rank, K),
            "head_frac": Met.group_fraction(topk, ctx.item_group, "head", K),
            "tail_frac": Met.group_fraction(topk, ctx.item_group, "long_tail", K),
        })

        explanations[label] = {}
        for i in recs[label]:
            try:
                e = model.explain(u, i)
            except Exception as exc:                       # keep the stage robust
                e = []
                print(f"  [warn] explain failed for {label}/{i}: {exc}")
            explanations[label][i] = {"raw": e, "text": reason_text(label, e, titles)}

        print(f"  {label:14s} NDCG@10={rows[-1]['ndcg_at_10']:.4f} "
              f"hits={int(Met.precision_at_k(topk, relevant, K) * K)}/10 "
              f"tail={rows[-1]['tail_frac']:.1f}")

    labels = [l for l, _ in PARADIGMS]

    # ---- Table 1: Model | Rank 1 .. Rank 10 ------------------------------------------
    rank_tbl = pd.DataFrame(
        [{"Model": lab, **{f"Rank {r + 1}": titles[recs[lab][r]] for r in range(K)}}
         for lab in labels])
    save_table("target_top10_by_model", rank_tbl)

    detail = pd.DataFrame([
        {"model": lab, "rank": r + 1, "item": recs[lab][r],
         "title": titles[recs[lab][r]],
         "train_popularity": int(ctx.item_pop[recs[lab][r]]),
         "popularity_group": ctx.item_group[recs[lab][r]],
         "held_out_rating": truth_map.get(recs[lab][r], np.nan),
         "is_hit": recs[lab][r] in relevant}
        for lab in labels for r in range(K)])
    save_table("target_top10_detail", detail)

    # ---- Overlap / consensus ---------------------------------------------------------
    counts: dict[int, list[str]] = {}
    for lab in labels:
        for i in recs[lab]:
            counts.setdefault(i, []).append(lab)
    consensus = pd.DataFrame([
        {"item": i, "title": titles[i], "n_models": len(v), "models": ", ".join(v),
         "train_popularity": int(ctx.item_pop[i]), "group": ctx.item_group[i],
         "held_out_rating": truth_map.get(i, np.nan)}
        for i, v in counts.items()]).sort_values(["n_models", "train_popularity"],
                                                 ascending=[False, False])
    save_table("target_consensus", consensus)
    # ---- Disagreements ---------------------------------------------------------------
    dis = disagreement_items(recs, K)
    dis["title"] = titles[dis.item.to_numpy()]
    dis["train_popularity"] = ctx.item_pop[dis.item.to_numpy()].astype(int)
    dis["group"] = ctx.item_group[dis.item.to_numpy()]
    dis["held_out_rating"] = [truth_map.get(int(i), np.nan) for i in dis.item]
    save_table("target_disagreements", dis)

    top_dis = dis.head(5).item.astype(int).tolist()

    # ---- Table 2: Movie | per-model reasoning ----------------------------------------
    reason_rows = []
    for i in sorted(counts, key=lambda x: (-len(counts[x]), -ctx.item_pop[x])):
        r = {"Movie": titles[i], "item": int(i),
             "n_models_recommending": len(counts[i]),
             "train_popularity": int(ctx.item_pop[i]),
             "popularity_group": ctx.item_group[i]}
        for lab in labels:
            if i in explanations[lab]:
                r[f"{lab} reasoning"] = explanations[lab][i]["text"]
            else:
                ranked_by = "not in this model's Top-10"
                r[f"{lab} reasoning"] = ranked_by
        reason_rows.append(r)
    save_table("target_reasoning", pd.DataFrame(reason_rows))

    # ---- Narrated disagreements ------------------------------------------------------
    narr = []
    for i in top_dis:
        who = counts[i]
        entry = {
            "item": int(i), "title": str(titles[i]),
            "train_popularity": int(ctx.item_pop[i]),
            "popularity_group": str(ctx.item_group[i]),
            "recommended_by": who,
            "not_recommended_by": [l for l in labels if l not in who],
            "held_out_rating": (float(truth_map[i]) if i in truth_map else None),
            "rank_by_model": {l: (recs[l].index(i) + 1 if i in recs[l] else None)
                              for l in labels},
            "score_percentile_by_model": {},
            "explanations": {l: explanations[l][i]["text"] for l in who},
        }
        for l in labels:
            S = models[l].score_all()[u]
            mask = np.ones(ds.n_items, bool)
            mask[tgt.R_profile[u].indices] = False
            cand = S[mask]
            entry["score_percentile_by_model"][l] = float(
                (cand < S[i]).mean() * 100.0)
        narr.append(entry)

    save_table("target_metrics", pd.DataFrame(rows))
    save_json("target_user", {
        "target": {"user_id": tgt.user_id, "internal_index": tgt.u,
                   "roll": tgt.roll, "mapping": tgt.mapping_explanation,
                   "n_visible": int(len(tgt.kept)), "n_hidden": int(len(tgt.hidden)),
                   "n_global_test": int(len(gtest)),
                   "n_relevant_heldout": len(relevant)},
        "metrics": rows,
        "consensus_items": consensus.to_dict(orient="records"),
        "disagreements": narr,
        "recommendations": {l: [{"rank": r + 1, "item": recs[l][r],
                                 "title": str(titles[recs[l][r]])} for r in range(K)]
                            for l in labels},
        "explanations": {l: {str(i): explanations[l][i]["text"] for i in recs[l]}
                         for l in labels},
    })

    print("\nconsensus (recommended by >= 3 models):")
    for _, rr in consensus[consensus.n_models >= 3].iterrows():
        print(f"  {rr.title[:48]:50s} {rr.n_models} models  [{rr.models}]")
    print(f"\ntop disagreements: "
          f"{[str(titles[i])[:38] for i in top_dis]}")
    return rank_tbl


if __name__ == "__main__":
    main()
