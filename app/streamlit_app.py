"""
Streamlit demo for the MovieLens 100K from-scratch recommender study.

Run with:   streamlit run app/streamlit_app.py

Everything on screen comes from the same artifacts the report is built from, so the demo
can never show a number that the experiments did not produce.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.service import RecommenderService

st.set_page_config(page_title="MovieLens 100K Recommender Lab",
                   page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading models…")
def get_service() -> RecommenderService:
    return RecommenderService()


def main() -> None:
    st.title("🎬 MovieLens 100K — five recommenders, built from scratch")
    st.caption("UBCF · IBCF · Regression-based neighbourhood CF · SLIM · Graph propagation "
               "— no recommender libraries, every algorithm implemented in NumPy/SciPy.")

    try:
        svc = get_service()
    except Exception as exc:
        st.error(f"Could not load models: {exc}")
        st.info("Run `python run_experiments.py` first to build the artifacts.")
        return

    ids = svc.user_ids()
    target = svc._target_id()

    with st.sidebar:
        st.header("Controls")
        default_idx = ids.index(target) if target in ids else 0
        user_id = st.selectbox("User ID", ids, index=default_idx)
        model = st.radio("Model", svc.available_models(), index=0)
        k = st.slider("Top-K", 5, 20, 10)
        show_all = st.checkbox("Side-by-side comparison of all models", value=False)
        st.divider()
        s = svc.user_summary(user_id)
        st.metric("Known ratings (train)", s["n_known_ratings"])
        st.metric("Mean rating", f"{s['mean_rating']:.2f}")
        st.caption(f"History group: **{s['history_group']}** · "
                   f"held-out test ratings: {s['n_heldout_test_ratings']}")
        if s["is_roll_target"]:
            st.success(f"User {user_id} is the roll-number target user.")

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.subheader("What this user already rated")
        prof = svc.user_profile(user_id)
        st.dataframe(
            prof[["title", "rating", "train_popularity", "popularity_group"]]
            .rename(columns={"title": "Movie", "rating": "Rating",
                             "train_popularity": "Popularity",
                             "popularity_group": "Group"}),
            hide_index=True, use_container_width=True, height=430)

    with right:
        st.subheader(f"Top-{k} from {model}")
        recs = svc.recommend(user_id, model, k)
        st.dataframe(
            pd.DataFrame([{"#": r["rank"], "Movie": r["title"],
                           "Score": round(r["score"], 4),
                           "Pred. rating": round(r["predicted_rating"], 2),
                           "Popularity": r["train_popularity"],
                           "Group": r["popularity_group"],
                           "Genres": r["genres"]} for r in recs]),
            hide_index=True, use_container_width=True, height=430)

    st.subheader("Why these recommendations?")
    for r in recs:
        with st.expander(f"{r['rank']}. {r['title']}  ·  {r['popularity_group']}"):
            st.write(f"**{r['explanation']['headline']}**")
            ev = r["explanation"]["evidence"]
            if ev:
                for e in ev:
                    st.write(f"- {e}")
            else:
                st.write("- no per-item evidence available for this model")

    if show_all:
        st.subheader("Side-by-side")
        st.dataframe(svc.compare(user_id, k), use_container_width=True)

    st.subheader("Measured test-set performance")
    st.caption("Every number below is read from results/tables/multiuser_test.csv, "
               "produced by the experiment pipeline on the held-out test split.")
    m = svc.model_metrics(model)
    if "note" in m:
        st.info(m["note"])
    else:
        cols = st.columns(5)
        for (kk, vv), c in zip(list(m.items())[1:11], cols * 3):
            c.metric(kk, f"{vv:.4f}" if isinstance(vv, float) else str(vv))
        st.caption(f"Configuration: `{m['spec']}`")
    allm = svc.all_model_metrics()
    if not allm.empty:
        st.dataframe(allm.round(4), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
