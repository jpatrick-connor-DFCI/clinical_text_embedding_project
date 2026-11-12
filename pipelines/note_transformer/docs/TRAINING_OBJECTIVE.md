# Global event-centered sampled Cox training

## Why sampled landmark-Cox instead of full time-varying Cox?

A longitudinal Transformer makes the patient representation itself an expensive
time-dependent computation. In a full event-time-updated Cox formulation, a
score may depend on the entire history available just before each failure time,
not on one reusable baseline vector. Naively encoding every at-risk patient's
history at each event can repeat large neural computations and retain large
activation graphs. A large risk set also increases input and denominator work.

Our scientific target is prediction at selected clinical landmarks: the model
uses information available by the chosen prediction occasion and ranks subsequent
first-event primary risk. We trade exhaustive case-anchor and denominator
evaluation for repeated stochastic exposure to prediction occasions and global
controls. Each eligible event appears once per epoch with one uniformly selected
pre-event anchor; at most a configured number of global controls is encoded.
Anchors and controls are redrawn in later epochs. Group-wise backpropagation makes
the event-group mean manageable without storing every group's activations at once.

There are two distinct design choices. Landmark conditioning fixes the information
available for the prediction task. Control sampling approximates the full-risk-set
denominator for that landmark-conditioned task. It does not recreate the exact
partial likelihood of representations updated at every event time. The bounded
work motivates this design; it does not establish empirical runtime, memory or
C-index superiority over an optimized full model.

## Mathematical construction and implemented semantics

Let `T_j` be patient `j`'s finite observed first-event/censoring day, let
`D_train` be the fixed patient-level training split, and let epoch be `e`.
For every observed primary-event case with a nonempty eligible-anchor set:

$$
\mathcal{L}_i=\lbrace t_0\in\mathcal{T}_{\mathrm{LM}}:t_0\lt T_i\rbrace,\qquad
t_{0i}^{(e)}\sim\mathrm{Uniform}(\mathcal{L}_i).
$$

If `L_i` is empty, the event is excluded and counted. Let a note embedding `x_jk`
be available at day `d_jk`. First filter notes to those available by the landmark:

$$
\mathcal{H}_j(t_0)=\lbrace (x_{jk},d_{jk}):d_{jk}\le t_0\rbrace.
$$

Sort this filtered history chronologically, breaking same-time ties by note ID,
and retain its latest at most `M` notes. Denote that ordered, capped history by
$H_j^M(t_0)$; history capping always follows landmark filtering. The score is

$$
\eta_j(t_0)=f_\theta(H_j^M(t_0),t_0,z_j).
$$

`z_j` denotes optional static covariates. Gaps are computed on this retained
history, with first gap zero. Empty histories use the learned landmark token.

$$
R(T_i)=\lbrace j\in\mathcal{D}_{\mathrm{train}}:T_j\ge T_i\rbrace,\qquad
N_i=|R(T_i)|-1,\qquad m_i=\min(m_{\mathrm{max}},N_i).
$$

Draw `S_i` uniformly among the size-`m_i` subsets of `R(T_i) \ {i}`,
without replacement, and define `R_tilde_i = {i} union S_i`. Crucially, every
member's history and score use `t0_i`, while membership uses the later actual
event time `T_i`:

$$
\ell_i^{(e)}=-\eta_i(t_{0i}^{(e)})+
\log\left(\sum_{j\in\widetilde R_i^{(e)}}
w_{ij}\exp(\eta_j(t_{0i}^{(e)}))\right).
$$

$$
L_B=\frac{1}{|B|}\sum_{i\in B}\ell_i^{(e)}.
$$

The numerical implementation evaluates the log denominator with stable
log-sum-exp applied to each score plus its log weight.

The case always has `w_ii=1`. Default control weights equal one. With optional
uniform inverse-inclusion weighting and `m_i > 0`, control weights are `N_i/m_i`.
The denominator estimate can be unbiased before its logarithm, while the log
partial-likelihood contribution is generally not exactly unbiased. This distinction
also applies to its stochastic gradients. A singleton group contributes zero.

The current source was verified against these six requirements without a core
code change:

| Requirement | Implementing path |
|---|---|
| Uniform strict pre-event landmark | `RiskSetSampler.sample`: uniform choice from anchors `< T_i`; constructor excludes empty sets |
| Landmark-truncated, recently capped history | `NoteData.history`: sorted timestamps, `searchsorted(..., side="right")`, cap after cutoff |
| Global actual-event-time risk pool | `RiskSetSampler.risk_indices`: sorted training-cohort times, inclusive `T_j >= T_i` |
| Uniform controls without replacement | `RiskSetSampler.sample`: `rng.choice(..., replace=False)` after removing the case |
| Same anchor for all group members | `train_epoch`: one `data.batch(group.member_ids, group.landmark, ...)` call |
| Stable weighted partial likelihood | `sampled_cox_loss`: finite positive weights, case weight one, `torch.logsumexp(scores + log(weights))` |

## Conceptual computational comparison

Let `N` be the training cohort size, `E` the eligible primary-event count,
`r_i=|R(T_i)|`, `K` the number of landmarks, and `C_enc(L)` the cost of one
Transformer history encoding. With note cap `M`, the model sees at most `M+1`
positions, including the landmark prefix. Actual costs depend on padding,
embedding dimension, hidden width, layers and attention backend.

| Strategy | Conceptual patient encodings | Neural work, using a common history-cost bound |
|---|---|---|
| Naive full event-time-updated risk sets | `sum_i r_i`, potentially order `E*N` | `sum_i r_i * C_enc(L)`; histories/scores may change at every event time |
| Naive full risk sets at sampled landmarks | `sum_i r_i` | Same denominator size; every member uses the group's chosen landmark |
| Exhaustive case-anchor pairs with full controls | `sum_i len(L_i) * r_i` | Up to order `E*K*N*C_enc(L)` without reuse |
| Implemented sampled landmark groups | `sum_i (1+m_i) <= E*(1+m_max)` | Bounded neural work per group; one anchor per case per epoch |

These counts describe naive evaluation with repeated encodings. For a fixed
landmark and fixed parameters, scores can be computed once for the eligible
cohort and reused by a full Breslow calculation; across `K` fixed anchors this
may require about `K*N` encodings instead. Caching graphs, exact reuse, stochastic
dropout and parameter updates introduce different compute/memory tradeoffs.
Specialized time-dependent encoders or incremental representations may also
reduce repeated work. Our package does not benchmark every such optimized method.

For classical dense attention, a rough per-history operation count includes
projection `O(L*D_input*d)` plus, per layer, attention `O(L^2*d)` and feed-forward
work `O(L*d*f)`, with additional hidden projections. This can dominate scalar
denominator arithmetic. Backend-specific attention kernels and inference versus
backpropagation change memory requirements; encoding-count ratios are not
guaranteed wall-clock or peak-memory ratios.

**Risk-set indexing is a separate cost.** Sorting observed times once is
`O(N log N)`, and each boundary search is `O(log N)`. The returned tail initially
shares storage, but removing the case materializes `r_i-1` candidate indices,
requiring order `r_i` work/storage. Anchor eligibility, case checks and NumPy's
sampling algorithm add overhead. Consequently, the current sampler does not
make all per-event work independent of cohort size, even though expensive
Transformer evaluations are bounded by the control cap. Likewise, history cutoff
lookup and dense-vector copying are distinct from neural computation.

Training backpropagates each group immediately, so `batch_size` accumulates a
group mean without multiplying simultaneously retained graphs by the number of
groups. One large full group can still be costly; full approaches can alternatively
microbatch, reuse common-landmark scores or apply other optimizations. Validation
already uses deterministic full eligible cohorts per landmark and the separate
full Breslow reference.

See [the optional benchmark](BENCHMARKING.md). It compares **naive full versus
bounded sampled evaluation at the same selected landmarks**, with fixed weights,
and labels its scope explicitly. Its two denominators yield different losses;
it supplies no accuracy comparison and does not implement a full event-time
time-varying Cox training baseline.

## Event-group training in practice

For every observed primary event in training patient `i` at actual day `T_i`:

1. Construct `L_i = {t0 in landmarks : t0 < T_i}`.
2. If empty, exclude the event from training and record the exclusion count.
3. Draw one `t0_i` uniformly from `L_i`.
4. Query **the entire training cohort** for `R(T_i) = {j : T_j >= T_i}`.
5. Include the case; sample up to `risk_set_max_controls` other members uniformly
   without replacement. Patients with no notes remain eligible controls.
6. Encode every member using its history through the **same** sampled `t0_i`.
7. Compute the case-first sampled partial-likelihood contribution.

```text
eta_j(t0) = model(history_j_through_t0, t0, optional_static_j)
loss_i = -eta_i(t0_i) + logsumexp_j(eta_j(t0_i) + log(w_ij))
L_batch = mean_i(loss_i)
```

This follows from the conditional event probability proportional to
`exp(eta_i)` among the sampled members, whose log denominator is evaluated using
stable `logsumexp`. All primary-event group losses have equal weight. The case
denominator weight is one. Default controls also have weight one. Optional
`use_ipw=true` gives sampled controls weight `N/m` when sampling `m` controls from
`N` eligible controls. This estimates the full denominator **before** the log;
Jensen's inequality means the log partial likelihood is not exactly unbiased.

**Worked numerical example:** if the case has relative risk 4 and two controls
have risks 2 and 1, the model scores are `log(4), log(2), log(1)` and the
unweighted sampled loss is

`-log(4) + log(4+2+1) = log(7/4) ≈ 0.560`.

`batch_size` counts event-centered groups, not ordinary patient minibatches.
The sampler's pool does not depend on which groups coexist in a training step.
Controls and anchors are redrawn from deterministic epoch-specific random streams.
The same patient may therefore appear at different landmarks across epochs.
Each event is weighted once per epoch, so eligible event-landmark pairs have
sampling weight `1/len(L_i)`; this deliberately differs from equally weighting
every expanded patient-landmark pair.

Tied failures use inclusive `T_j >= T_i` risk membership, including tied censoring
and other tied cases. With all controls at one fixed landmark, the sum of these
contributions is the Breslow partial likelihood. There is no Efron correction.
`full_breslow_loss` implements an independent full-cohort reference using sorted
tie blocks and stable cumulative log sums. Singleton groups contribute zero and
are counted; a training population with only singleton groups is rejected.

This is first-event cause-specific landmark Cox, **not** ordinary minibatch Cox,
general recurrent-event Andersen–Gill, or Fine–Gray modeling. It assumes reliable
first-event/right-censoring observations, cohort availability at modeled landmarks,
and a meaningful landmark-relative proportional-hazards risk score. Delayed entry
and interval censoring are not supported. Informative censoring, observational
selection and note-generation patterns require clinical/statistical review.

## Hazard formulation and score interpretation

At a fixed landmark, the cause-specific hazard can be written as

`h(u | H(t0), t0) = h0(u; t0) * exp(eta(H(t0), t0))`,

where `u` is follow-up time after the landmark. The unspecified baseline cancels
from the conditional partial-likelihood ratio. For the sampled denominator,
`p_i = exp(eta_i) / sum_j(w_ij * exp(eta_j))`; minimizing `-log(p_i)` gives the
implemented contribution. With IPW weights, this is a denominator approximation,
not a calibrated event probability or an exactly unbiased full log likelihood.

The head returns unconstrained log-relative-risk, with larger values indicating
higher modeled primary-event hazard. There is no fitted baseline hazard, absolute
survival curve or cumulative-incidence curve. C-index is evaluated separately
at landmarks; cross-landmark score offsets are not identified. Competing death
removes a patient from later primary-event risk sets, with no separate death head.

Implementation: `risk_sets.py`, `losses.py`, and `train.train_epoch`. Validation
uses the independent full-cohort `full_breslow_loss`, event-normalized; its scale
should not be compared directly with sampled training loss.
