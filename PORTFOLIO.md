# Interview preparation pack

Technical questions this project invites, with answers grounded in the measured results. The executive summary, the CV bullets and the 30-second pitch live in [README.md](README.md) and are deliberately not repeated here.

One-line context: five recommendation paradigms implemented from scratch on MovieLens 100K (943 users x 1682 movies, 100,000 ratings, 93.70% sparse); best ranker SLIM at NDCG@10 0.2950 against 0.1535 for a popularity baseline; best RMSE 0.9191 against 0.9438 for a regularised bias baseline.

---

## Likely technical interview questions, with answers

**1. Why did you compute Pearson correlation with matrix products instead of a loop over user pairs?**

Textbook co-rated Pearson needs five sums restricted to the intersection `I_a ∩ I_b`. Multiplying by the binary indicator `B` *is* that restriction, so `n = B Bᵀ`, `S_a = R Bᵀ`, `S_ab = R Rᵀ` and `Q_a = R² Bᵀ` give every pair's sums in five sparse GEMMs. The naive loop is O(n² m) in Python; this is BLAS-bound and exact — I verified it against a brute-force implementation to 0.0 absolute error. The general lesson is that 'restrict a sum to a set' is usually a multiplication by an indicator, and that turns a loop into a GEMM.

**2. You said you use the *exact* per-(u,i) neighbourhood. Why does that matter?**

The common shortcut is to take user u's global top-k neighbours once and then keep whichever of them rated item i. For a popular item that is fine; for an unpopular item almost none of the k neighbours rated it, so the effective neighbourhood silently shrinks to two or three users. That makes k-curves look better than they are and hides exactly the failure mode you care about — unpopular items. I recompute the top-k among `U_i` for every item, which costs `|U| × nnz` element operations (about 6.7e7 here) done as 1,682 vectorised argpartitions.

**3. Your neighbourhood models rank by something other than predicted rating. Why?**

Because ranking by `rhat` is dominated by items with almost no evidence. One enthusiastic neighbour who rated an obscure film 5 produces `rhat = 5.0` with a denominator of a single similarity. So each model exposes three heads — `rating`, unnormalised `score`, and implicit `affinity` — and the choice is made on validation. Measured on this data the rating head reaches NDCG@10 0.0127 for UBCF while the selected head reaches 0.1813. That gap is the quantitative form of the 'never rank by predicted rating' rule.

**4. What exactly does the regression-based neighbourhood model learn that similarity does not?**

Similarity `s(i,j)` is a *marginal* association — how j relates to i ignoring everything else. The ridge solution `w = (ZᵀZ + λI)⁻¹Zᵀy` gives *partial* coefficients — how j relates to i holding the other neighbours fixed. So near-duplicate neighbours stop being double-counted, and a neighbour can even take a negative weight despite a positive similarity (a suppressor effect); 1.9% of coefficients do exactly that. Also, similarity weights are normalised to sum to 1 by construction, so they always produce a full-strength prediction; learned weights sum to 0.805 on average and can therefore say 'this neighbourhood barely determines the target'.

**5. Did learning the weights actually help? Be honest.**

Yes, and less than you might hope. Validation RMSE 0.9180 learned versus 0.9212 with similarity weights in the identical architecture — real, but a third of a percent. And it is entirely the regularisation doing the work, not the learning. At the same neighbourhood size the unregularised fit reaches 1.2077 against 0.9180 for the best ridge setting, and it degrades as k grows (k=80 gives 1.3651, worse than predicting the global mean at 1.1203) because the design matrix has k columns and only as many rows as there are users who rated the target item. You can watch the shrinkage happen: the correlation between the learned coefficients and the similarities they replaced rises monotonically with λ, from 0.01 at λ=0 to 0.71 at the top of the grid — the regression is being pulled back onto the heuristic it was meant to improve, and the useful setting is the one in between.

**6. Why FISTA for SLIM rather than coordinate descent, which is the usual choice?**

Both work; the question is what dominates. Coordinate descent on the Gram matrix needs a rank-1 update of an n×n matrix per coordinate, so a full sweep is O(n³) in memory-bandwidth-bound operations. FISTA needs one dense GEMM `G W` per iteration — the same O(n³) flops, but as a BLAS-3 call running one to two orders of magnitude faster per operation. Since `W ≥ 0`, the L1 penalty is linear on the feasible set and its proximal operator collapses to `max(0, V - η λ₁)`, so the whole non-smooth part is one shift and a diagonal zeroing. Nesterov momentum gives O(1/t²). It converges here in 175 iterations.

**7. How do you know your SLIM solution is actually optimal and not just stopped?**

A stalled objective is necessary but not sufficient. I check the KKT conditions of the non-negative elastic net directly: with `g = GW - G + λ₂W`, an active coefficient needs `g_ij + λ₁ = 0` and an inactive one needs `g_ij + λ₁ ≥ 0`. The delivered model stops at a relative violation of 9.67e-05 — absolute 0.0350 against a Gram scale of 362. That certificate is computed from G and W alone, so it is independent of the solver that produced them, and the solver stops on it rather than on the objective.

**8. How did you pick the implicit-feedback threshold, and why not just use 4?**

I swept it. τ ∈ {1,2,3,4,5} at the selected regularisation, on validation: τ=1 → NDCG@10 0.1869, τ=2 → NDCG@10 0.1971, τ=3 → NDCG@10 0.2058, τ=4 → NDCG@10 0.2081, τ=5 → NDCG@10 0.1552. The measured argmax is τ = 4, which happens to be 4 — but 'happens to be' is the point. Both directions degrade for predictable reasons: τ=1 admits 1-star ratings as endorsements and pollutes the co-occurrence counts, τ=5 starves the Gram matrix of support. The threshold is a hyper-parameter, not a convention.

**9. Why is your graph recommender biased toward popular items?**

It isn't a data artefact — it's the operator. The stationary distribution of a random walk on an undirected graph is proportional to node degree. So as you weaken the restart or lengthen the paths, personalised PageRank relaxes toward the degree distribution, which is exactly popularity. The measurements track the theory: corr(path length, head-item share) = 0.717, corr(restart α, mean popularity rank) = 0.973. Degree normalisation (`D⁻¹A` instead of `A`) is the built-in mitigation, because it makes each node spread one unit of evidence rather than an amount proportional to its degree.

**10. How did you avoid data leakage?**

Four separate guards. (1) `fit` only ever receives `R_train`, so every similarity, bias, Gram matrix, learned weight and graph edge comes from train. (2) Candidate sets exclude only *training* items — excluding test items would hand the model the answer key by shrinking the candidate pool to the right answers. (3) Every hyper-parameter is chosen on validation; the test split is read once, in one stage. (4) Even the analysis strata — popularity groups, novelty weights, user-history tertiles — are computed from training counts, so the way I slice the results is leakage-free too. The gate suite re-checks the split is an exact disjoint partition and that no recommended item appears in the user's training profile.

**11. Your best ranker has a mediocre RMSE and your best RMSE model ranks poorly. What's going on?**

They optimise different things and the data lets you see it. RMSE is an average over *observed* held-out ratings, most of which sit in the middle of the scale; ranking quality is decided entirely by the extreme top of a 1,682-item ordering. A model can be beautifully calibrated in the middle and order the tails badly. The error analysis shows the mechanism: every model over-predicts low ratings and under-predicts high ones, because squared-error training pulls predictions toward the conditional mean and narrows the predicted distribution. Concretely here, IBCF reaches RMSE 0.9402 with NDCG@10 0.2593, while SLIM reaches NDCG@10 0.2950 with RMSE 1.0331.

**12. Which model would you actually deploy?**

SLIM for the head of the list, with a second stage underneath it. It is the strongest ranker, its learned matrix is 93.9% sparse so serving is a sparse lookup with no user-side computation, it retrains in 22.0 s on this data, and every slot is explainable by one coefficient. The caveat is not optional: on its own it reaches 20.6% of the catalogue and draws 1.7% of its slots from the long tail. Shipping it alone quietly shrinks the catalogue to a few hundred movies, which is a business problem long before it is a metrics problem.

**13. What breaks if I hand you 100 million ratings instead of 100 thousand?**

The algorithms survive; these implementations do not, and I'd rather be specific about where. Dense similarity matrices are O(|I|²) — 22 MB here, 80 GB at 10⁵ items. SLIM's dense Gram has the same wall, and the fix is column-wise coordinate descent over a top-M co-occurrence candidate set, which is embarrassingly parallel and O(|I|·M). Interestingly, I measured that feature-restriction trick here and it made training *slower*, because at 1,682 items the dense GEMM already dominates and the masking is pure overhead — it only pays once the dense Gram stops fitting. UBCF is the worst-scaling of the five in practice, because its model is O(|U|²) and |U| is the dimension that grows. Item-item and SLIM scale best: the model is bounded by catalogue size and can be precomputed offline.

**14. Why intra-list diversity over genres rather than over your own similarities?**

Because measuring diversity with the same similarity a model optimises makes every model look diverse by construction — the metric and the objective would share a failure mode. The 19 genre flags are an external, model-independent yardstick. It is coarse, and I say so in the limitations; the alternative is circular.

**15. How would you know any of this generalises? You ran one split.**

I wouldn't, fully, and that's stated as limitation 2. The `*_std` columns are across-user standard deviations, not confidence intervals over repeated splits, so differences below roughly 0.005 NDCG shouldn't be over-read. The honest fix is repeated seeded splits with bootstrap confidence intervals over users and paired tests between models, which is the first item on the future-work list. What does carry weight is the *ordering* on the qualitative axes — accuracy versus coverage, propagation depth versus popularity — because those have mechanisms behind them, not just point estimates.

**16. Which method is best for cold-start users?**

On this data the sparse-history tertile (≤ 30 training ratings, n = 321) is best served by SLIM on ranking (NDCG@10 0.2640) and RegressionCF on rating error (RMSE 0.9863). The model that gains most from a richer history is RegressionCF (+0.1477 NDCG@10 from sparse to heavy). Note the framing though: MovieLens guarantees every user at least 20 ratings, so there are no truly cold users here. The real sparsity problem in this dataset is on the item side, where the median movie has 27 ratings.

---

*All figures quoted above are read from `results/` by `src/report.py`; none are typed by hand. Seed 2310049, derived from roll number 23IM10049.*
