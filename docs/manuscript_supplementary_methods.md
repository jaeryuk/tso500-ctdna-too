# Supplementary Methods

Detailed procedures supporting the Methods. Section numbers are referenced from the main text.

> Internal markers: `[VERIFY]` = number carried from the Results draft, not re-derived from code.
> `[PENDING]` = analysis not yet executed.

---

## S1. Cohort resolution

Rule-conformant manifests key samples by TSO identifier (e.g. `TSO_00048_B_01`) with a `tumor_type1`
label and a `keep_t1` inclusion flag, whereas feature matrices key samples by a 10-digit numeric
identifier. The two are reconciled through the instrument rename log, and only `keep_t1` rows that
resolve to a numeric identifier **and** carry a non-empty `tumor_type1` are retained. The per-version
minimum class size is 20. This procedure is implemented once and shared by every downstream script so
that no figure can silently use a different sample set.

The resulting cohort (v1 1,093 / 10 types; v2 796 / 8 types) differs from an earlier
"keep_t1, minimum class size 8" cohort used during exploratory work; results computed on the earlier
cohort are not reported.

## S2. Fragment-level extraction

### S2.1 Windowing and I/O

Binding sites are grouped by chromosome and clustered when separated by less than twice a margin
(200 bp), so each cluster is fetched once and profiles for all member sites are sliced from a single
coverage vector. Per site, a base-resolution endpoint profile is built over ±160 bp (321 positions,
centre index 160) by incrementing at both fragment endpoints.

### S2.2 Strand orientation and window geometry (exact indexing)

For a plus-strand site, the oriented ±40 bp window is taken directly as profile indices
`[EXT−40, EXT+40)`, i.e. genomic offsets **−40 … +39** relative to the motif centre. For a
minus-strand site, the **full 321 bp profile is reversed first** and then the same slice is taken;
because the centre sits at the midpoint of an odd-length vector, reversal maps the centre onto itself,
and the oriented window again spans offsets −40 … +39 (corresponding to genomic +40 … −39).

This detail matters whenever a *predicted* profile must be compared with the observed one: the
prediction must be sliced as **plus: c−40 … c+39; minus: c−39 … c+40 reversed**. Reversing an 80 bp
slice instead of the full profile introduces a 1 bp phase offset on minus-strand sites, shifting the
5 bp bin boundaries for approximately half of all sites.

### S2.3 Definitions in full

Endpoint Gini uses all qualifying fragments (TLEN 1–800), 16 bins of 5 bp, as in the main text.
Fragment-length entropy uses fragments whose **midpoint** lies within ±40 bp of the centre, lengths
30–420 bp in 5 bp bins (78 bins), Shannon entropy in bits. Both are `NaN` where the relevant count
is zero; missing values are handled at matrix-assembly time by per-cohort column-median imputation.

Note that a "length entropy" computed instead over fragments *overlapping* the window, using
per-unique-length counts and natural logarithm, appears in some interpretive/plotting utilities; the
canonical feature used for modelling is the midpoint/5 bp/bits definition given above.

### S2.4 Support statistics

`nep` is the endpoint count inside the oriented ±40 bp window; `nfr` is the fragment count entering
the length-entropy calculation. For the target sample used in exemplar figures, `nep` had median 648
(10th–90th percentile 229–1,186).

## S3. Support dependence of the Gini statistic

The Gini coefficient of a 16-bin histogram is not support-free: with finite counts it is inflated by
sampling noise. Under a **uniform null with no biological structure**, simulated multinomial draws give

| endpoints *n* | E[Gini] under uniform null |
|---:|---:|
| 50 | 0.293 |
| 100 | 0.210 |
| 229 | 0.139 |
| 400 | 0.106 |
| 648 | 0.083 |
| 1,186 | 0.061 |
| 2,000 | 0.047 |
| 5,000 | 0.030 |

Empirically, per-site endpoint Gini correlates with support at Spearman ρ ≈ −0.49 (≈24% of variance),
and the sequence-bias adjustment of §S4 does not reduce this (ρ ≈ −0.50), because the simulated Gini
is computed from a continuous expected density and therefore carries no sampling noise.

Two mitigations were considered. The `nep > 300` site filter (Methods) removes the lowest-support
sites and is applied inside the training partition. A ranking metric residualised on log-support —
OLS of $G_s$ on $\log(\text{nep}_s)$, retaining the residual — is used in one interpretive panel;
note this is a linear detrend of a non-linear relationship and, more importantly, removes *all*
support-correlated variation including any genuine biology, so it is used only for site ranking and
never as a model feature.

`[Count-matched normalisation — i.e. subtracting E[Gini | n, p] under a multinomial null — was
evaluated and is out of scope for this manuscript by author decision.]`

## S4. Sequence-bias model and bias-adjusted Gini

### S4.1 Bias model

Cut contexts are sampled from observed fragment ends; the 3′ end context is reverse-complemented so a
single matrix applies to either end. Observed context base frequencies are compared with a
**position-matched genomic background** — for each observed cut at position *p*, a background position
is drawn uniformly within ±200 bp of *p* — and the model is the elementwise log-odds

$$
\text{lo}[j, b] \;=\; \log \frac{f^{\text{obs}}_{j,b}}{f^{\text{bg}}_{j,b}}, \qquad j = 0,\dots,2c_k-1,\; b \in \{A,C,G,T\},
$$

with pseudocount 1 in both tables. With half-width $c_k = 10$ the context spans 20 bp; fitted models
have max |log-odds| 0.296 (v1) and 0.338 (v2), concentrated at offsets −3 … +1 relative to the cut.

### S4.2 Per-site cut propensity

Applying the model to the reference at each site gives, for window position *p*,

$$
w^{+}_p = \exp\!\Big(\textstyle\sum_j \text{lo}\big[j,\; \text{base}(p - c_k + j)\big]\Big),
\qquad
w^{-}_p = \exp\!\Big(\textstyle\sum_j \text{lo}\big[j,\; \overline{\text{base}(p - j + c_k)}\big]\Big),
$$

where $w^{+}_p$ is the propensity that a fragment's **first** base is *p*, $w^{-}_p$ that its **last**
base is *p* (overbar = complement), and positions overlapping an `N` receive weight 0.

### S4.3 Expected endpoint density and the simulated Gini

With $\text{pmf}(L)$ the sample's fragment-length distribution, the expected endpoint density at
position *k* is

$$
D_k \;=\; w^{+}_k \sum_{L} \text{pmf}(L)\, w^{-}_{k+L-1} \;+\; w^{-}_k \sum_{L} \text{pmf}(L)\, w^{+}_{k-L+1},
$$

the first term counting *k* as a fragment start and the second as a fragment end. $D$ is binned on the
identical ±40 bp/5 bp geometry (with the strand phase of §S2.2) and $G^{\text{sim}}$ is its Gini;
because Gini is scale-invariant, no normalisation of $D$ is required. Evaluating $D$ over ±40 bp with
lengths up to 520 bp requires $w^{-}$ out to *c*+559 and $w^{+}$ back to *c*−559, so weights are
precomputed over a symmetric ±560 bp window.

Two properties are worth recording. First, the length distribution is **immaterial**: the ~520 bp
convolution nearly flattens both sums, so $D \approx w^{+} + w^{-}$ up to scale. Substituting the
sample's scanned PMF (mode 166 bp) with a narrow 160–175 bp PMF changes the predicted profiles by a
median correlation of +0.992 (minimum +0.969). Second, a 5′-only approximation (using $w^{+}$ and a
length convolution in place of $w^{-}$) understates the sequence contribution: on the exemplar sample
median $G^{\text{sim}}$ rises from 0.096 to 0.128 when the 3′ term is included.

### S4.4 Bias-control benchmark

The three arms (observed / bias-adjusted / sequence-only) are compared as full-dimensional elastic-net
models over the 26,845-site feature space under the same 5 × 20 nested cross-validation, with boxes
formed from the 20 outer repeats. `[VERIFY: the current bias-control figure was computed on the earlier
cohort (v1 n = 1,049; v2 n = 759 with K = 7) and must be refitted on the rule-conformant cohort before
submission; usable n after requiring both raw and adjusted extractions is v1 1,005 / v2 718 unless the
remaining samples are extracted.]`

## S5. Motif-resolved sequence preference of fragment endpoints

To test whether individual motifs carry their own endpoint sequence preference, the sequence-predicted
endpoint profile of §S4.3 was compared per TF with the observed profile over the oriented ±40 bp
window, both per-site sum-normalised, for TFs with ≥ 20 sites (176 TFs).

- Matched pairing (each TF against its own motif): median Pearson *r* = **0.496**.
- Mismatched pairing (each TF against another TF's motif, 20 permutations): median *r* = **0.008**.
- 86.4% of TFs exceed the 95th percentile of the mismatched null.

Sequence preference is therefore genuinely motif-specific, and it is captured by the generic model
purely because each site's own sequence is the input — no TF-specific bias model is required. Its
magnitude is substantial (predicted max/min density up to ≈5×) and varies greatly by factor
(RELA, ERG, STAT3, CEBPB *r* ≈ 0.85–0.90; USF2, SP2, NR2C2, E2F1 *r* ≈ 0.03–0.08).

Crucially, sequence preference does **not** generate the motif-centred dip: predicted dips are flat for
every TF (median 0.994, range 0.939–1.084) whereas observed dips span 0.629–1.342, and across TFs the
two are uncorrelated (*r* = +0.025, *P* = 0.74). Sequence-explained profile shape is also unrelated to
dip depth (Spearman +0.054, *P* = 0.48).

## S6. Exemplar selection for single-sample figures `[VERIFY]`

Single-sample panels (endpoint-Gini versus motif-dip; Gini-stratified fragmentation profiles) use
sample 2513017372 from v2. **This sample was selected as the most-negative-ρ sample in v2** and sits at
the **0.5th percentile** of the v2 distribution (its ρ = −0.184 against a v2 median of −0.156;
per-sample ρ: v1 median −0.153, IQR [−0.164, −0.141], range [−0.223, −0.080]; v2 median −0.156,
IQR [−0.166, −0.140], range [−0.187, −0.096]).

The selection rule must be stated in the figure legend, and the cohort-wide distribution (not the
exemplar's value) should carry the claim in the main text; otherwise the panel reads as typical when
it is extremal. `[VERIFY: the Results text reports ρ = −0.166 for this panel and per-sample medians of
−0.128 (v1) / −0.134 (v2); the per-sample table gives medians of −0.153 (v1) / −0.156 (v2). The site
universe behind each number must be reconciled — the figure applies a cohort `nep > 300` mask, which
the per-sample table may not.]`

## S7. Nested cross-validation — implementation detail

Outer folds are `RepeatedStratifiedKFold(n_splits = 5, n_repeats = 20)` with a fixed seed, shared
across every module and ablation so that all comparisons are paired on identical folds. Within each
outer-training set, meta folds are `StratifiedKFold(5, shuffle = True)` seeded per outer fold. Inner
tuning is `GridSearchCV(cv = 3, scoring = "roc_auc_ovr")`.

Elastic-net grid: `l1_ratio` fixed at 0.5 (regularisation strength is the dominant knob and fixing the
mixing parameter reduces compute threefold); *C* tuned over the pre-specified grid.
XGBoost grid: `max_depth` ∈ {3, 4}, `learning_rate` ∈ {0.03, 0.1}, `n_estimators` ∈ {300, 600}.
Elastic-net models use `class_weight = "balanced"`; XGBoost is left unweighted so that its
probabilities remain well calibrated as combiner input. Convergence is tracked (`n_iter_` captured and
the converged fraction reported) and warnings are not globally suppressed. Phase-1 base predictions are
pickled per cohort so that a multi-day run survives restarts.

With 20 repeats, normal-approximation intervals use *t* = 2.093.

## S8. Shapley computation

For five modules the exact computation enumerates all 32 subsets. Within each subset the base-model
out-of-fold probabilities are fixed and only the NNLS combiner is refit, so $\phi_i$ measures a
module's contribution *given the fusion machinery*, not the cost of retraining. The efficiency identity
$\sum_i \phi_i = v(N) - v(\varnothing)$ is asserted numerically at every cohort × bin cell and the
computation aborts on violation.

For the tumour-fraction-stratified variant, a bin's evaluable classes are those present but not
saturating within the bin, i.e. $0 < |\{i \in \text{bin}: y_i = k\}| < |\text{bin}|$. In v2's 10–20%
bin (*n* = 83) one of eight classes is absent, so that point is macro-averaged over seven classes;
all other cohort × bin cells use the full class set.

## S9. GRAIL cohort — metadata `[PENDING]`

EGA dataset EGAD00001005302. The analysis1 (raw, uncollapsed) cfDNA set comprises 198 samples from 198
unique subjects: prostate cancer 54 (27.3%), lung cancer 49 (24.7%), breast cancer 48 (24.2%),
non-cancer 47 (23.7%).

Sex composition: breast 48 F / 0 M; prostate 0 F / 54 M; lung 32 F / 17 M; non-cancer 24 F / 23 M
(overall 104 F / 94 M). The complete sex separation of the breast and prostate classes is a structural
confounder of this dataset and is addressed in the main Methods.

Note that the labelled manifest's BAM path column points to the **analysis2 (collapsed)** alignments
(`*collapsed.bam`) whereas the fragmentomic analysis requires the **analysis1 (raw)** alignments
(`*raw.bam`); joins between the manifest and the raw download must therefore be made on the sample
stem rather than on the file path.

Download status as of 2026-07-15: 39/198 complete, 13 in flight, 146 queued — non-cancer 21/47 (44.7%),
lung 7/49 (14.3%), prostate 7/54 (13.0%), breast 4/48 (8.3%). The skew toward controls reflects
accession ordering; cancer-versus-control analyses become feasible before tissue-of-origin analyses.

## S10. Software

Python 3.11 (NumPy, pandas, scikit-learn, XGBoost, SciPy, pysam), R 4.x (ggplot2, ggpubr).
ichorCNA for tumour-fraction estimation. UniBind for direct TF–DNA binding sites. For the transcription-factor
occupancy / chromatin-accessibility validation of the cfDNA fragmentomic channels (endpoint Gini and
fragment-length entropy), transcription-factor footprints were taken from the ENCODE DNase-seq genomic-footprint
call set generated by the reference footprinting pipeline of Vierstra et al. (Global reference mapping of human
transcription factor footprints. *Nature* 2020;583:729–736), restricted to primary hematopoietic lineages and
pooled across experiments into a per-site footprint call frequency used to define bound versus unbound sites.
Reference GRCh37/b37 (`human_g1k_v37.fasta`). `[VERIFY: pin exact versions before submission.]`
