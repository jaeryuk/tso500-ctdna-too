# Methods

> **Status markers for internal use — remove before submission.**
> `[PENDING]` marks text describing analyses that have **not yet been executed** (GRAIL replication).
> `[VERIFY]` marks numbers taken from the Results draft that were not re-derived against the code.

---

## Patients, samples and ethics

Plasma samples were profiled by error-corrected targeted sequencing with the TruSight Oncology 500
(TSO500) assay (Illumina), which tiles 523 cancer-related genes. Two assay versions were used during
different clinical periods and are treated throughout as separate cohorts: v1 (*n* = 1,093, 10 cancer
types) and v2 (*n* = 796, 8 cancer types), giving 1,889 samples in total. `[VERIFY: ethics approval
number, consent statement, and institutional review board to be inserted.]`

### Cohort definition (rule-conformant cohort)

A single, pre-specified inclusion rule was applied to both versions, and every analysis in this paper
uses the resulting *rule-conformant* cohort. A sample was retained if it (i) appeared on the clinical
case list, (ii) was the primary blood draw (`B_01`), (iii) had median exon coverage ≥ 1,000×, and
(iv) carried a primary tumour-type label (`tumor_type1`). Cancer types with fewer than 20 samples
**within a version** were dropped from that version, because macro-averaged metrics are unstable in
very small classes. Where a subject contributed multiple sequencing runs, the latest run was kept, so
each sample corresponds to one patient.

This yields 1,093 v1 samples across 10 cancer types and 796 v2 samples across 8 cancer types. Seven
types (lung, colorectal, gastric, pancreatic, biliary tract, melanoma and prostate cancer) occur in
both; liver cancer, sarcoma and bladder cancer are v1-only, and breast cancer is v2-only.
`[VERIFY: the Results text states "seven cancer types ... represented in both cohorts" in the study-design
section but "the six cancer types shared between v1 and v2" in the cross-assay section. These must be
reconciled; the number used for transfer validation should be stated once and used consistently.]`

## Sequencing, alignment and fragment definition

Reads were aligned to the GRCh37/b37 human reference (`human_g1k_v37.fasta`). All fragment-level
analyses used a single, uniform filter: the read-1 mate of properly paired alignments, excluding
secondary, supplementary, duplicate and vendor-QC-failed records, with mapping quality ≥ 20 and
inferred template length 1–800 bp. A fragment was represented by its two genomic endpoints — the
first base (`reference_start`) and the last base (`reference_start + TLEN − 1`) — and by its length
(TLEN). Because both endpoints of every qualifying fragment are used, endpoint-based statistics are
not restricted to any fragment-length sub-population.

## Feature modules

Five complementary modules were extracted per sample. Feature construction, model fitting and
internal validation were performed **separately for v1 and v2** throughout; the two versions were
never pooled.

### TFBS SHAPE

**Site universe.** Direct TF–DNA binding sites with motif-resolved coordinates were taken from the
UniBind catalogue and intersected with the TSO500 target intervals. Only sites whose motif lies
entirely within a target interval were retained ("edge0"), giving **26,845 unique binding sites
spanning 252 transcription factors**. Sites are indexed identically across all samples, so a given
feature column refers to the same genomic locus in every sample.

**Definition.** For each site *s* we take a ±40 bp window around the motif centre, oriented to the
motif strand, and compute two quantities. We refer to the pair collectively as **S**ite-directed
**H**igh-resolution **A**nalysis of **P**ositional **E**ndpoints and **E**ntropy (SHAPE).

*Endpoint Gini.* The 80 bp window is divided into 16 bins of 5 bp. Let $c_{(1)} \le \dots \le c_{(B)}$
($B = 16$) be the sorted per-bin counts of fragment endpoints and $N=\sum_b c_b$. The Gini coefficient is

$$
G_s \;=\; \frac{2\sum_{b=1}^{B} b\, c_{(b)}}{B\,N} \;-\; \frac{B+1}{B},
$$

defined only where $N > 0$. $G_s = 0$ denotes endpoints spread uniformly across the window; larger
values denote a more unequal spatial distribution. The statistic is scale-invariant and, by
construction, does **not** indicate *where* endpoints concentrate — that is resolved separately by the
motif-dip ratio (below).

*Fragment-length entropy.* Fragments whose midpoint lies within ±40 bp of the motif centre are binned
by length into 5 bp bins spanning 30–420 bp (78 bins). With $p_j$ the fraction of fragments in bin
$j$, the Shannon entropy in bits is

$$
H_s \;=\; -\sum_{j: p_j>0} p_j \log_2 p_j .
$$

**Support and site filtering.** Per site we record the endpoint count $N$ in the ±40 bp window
(`nep`) and the fragment count contributing to $H_s$ (`nfr`). The SHAPE module used in the ensemble
(`SHAPE_nep300`) is restricted to sites whose **cohort median `nep` exceeds 300**. Critically, this
median is computed **within the cross-validation training partition only** and applied to the held-out
fold, so the site filter cannot leak outer-test information. In v1 this retains 24,768 of 26,743
evaluable sites.

**Feature matrix.** The SHAPE design matrix concatenates the per-site Gini and per-site length-entropy
channels, so a sample is represented by $2 \times n_\text{sites}$ features.

### Motif-dip ratio (interpretation only)

To establish what endpoint inequality reflects spatially, we computed, per site, the ratio of endpoint
counts in the motif centre to those in the flanks, $r_s = m_s / f_s$, and related it to $G_s$ by
Spearman correlation across sites within a sample. This quantity is used for biological interpretation
and is **not** an input to any classifier.

### Comparator modules

*Exon 1 fragment-length entropy* — fragment-length entropy computed over first-exon intervals,
following the published approach for transcription-associated cfDNA fragmentation.
*All-exon depth* — per-target normalised sequencing depth across all assay exons.
*Broad copy-number alterations* — arm-level and >10 Mb-scale log₂ coverage ratios derived from
on- and off-target reads.
*Somatic mutation profile* — per-gene somatic alteration indicators from the clinical variant output.
`[VERIFY: each comparator module needs its own normalisation/QC sentence written from its build script
before submission; they are summarised only briefly here.]`

## Tumour fraction

Tumour fraction was estimated from off-target reads with ichorCNA. Samples were stratified into four
pre-specified absolute bins — <3%, 3–10%, 10–20% and >20% — shared across both versions so that bins
are directly comparable between v1 and v2. Bin occupancy was v1 479/310/119/185 and v2 322/225/83/166.

## Model training and validation

### Base learners

Each module was modelled independently within each version. The primary learner is multinomial
elastic-net logistic regression implemented as a `StandardScaler → LogisticRegression(penalty =
"elasticnet", solver = "saga", class_weight = "balanced")` pipeline, so that scaling is refit inside
every training fold. The `l1_ratio` was fixed at 0.5 and the regularisation strength *C* tuned;
gradient-boosted trees (XGBoost) were used for modules where they were pre-specified as the better
learner, and were left unweighted to preserve probability calibration for the downstream combiner.

### Three-level nested cross-validation

All reported performance comes from a fully nested, leakage-controlled scheme:

1. **Outer** — `RepeatedStratifiedKFold` with 5 splits × 20 repeats. Outer-test folds are scored and
   are never touched by any fitting step.
2. **Meta** — within each outer-training set, `StratifiedKFold` (*K* = 5) cross-fits the base learners
   to produce out-of-fold probabilities used to fit the combiner.
3. **Inner** — within each meta-training fold, `GridSearchCV` (3-fold, scored by one-vs-rest
   macro-AUROC) tunes hyperparameters.

Base learners are then refit on the full outer-training set and applied to the untouched outer-test
fold, to which the **frozen** combiner is applied. Consequently no component — base model,
hyperparameter, feature filter or combiner weight — is informed by any outer-test sample.

### Late-fusion ensemble

Module probabilities were integrated by a per-class one-vs-rest non-negative least-squares combiner
with shrinkage (a "shrinkage NNLS super-learner"). For class *k*, with $Z_k$ the matrix whose columns
are the modules' predicted probabilities for class *k* and $t_k$ the binary class indicator, weights
solve

$$
\hat{w}_k \;=\; \arg\min_{w \ge 0} \; \big\| \, S^{1/2}\big([Z_k \; \mathbf{1}]\,w - t_k\big) \big\|_2^2 ,
$$

where $S$ balances the two classes (each contributing total weight ½). Fused probabilities are
renormalised across classes. The shrinkage constant was pre-specified at 60 from prior work, with
{30, 60, 100} carried as a sensitivity analysis. Combiner ablations (best-single, equal-weight,
global NNLS, per-class without shrinkage, leave-one-module-out) were run on identical folds.

### Metrics

Macro-AUROC is the primary metric; weighted AUROC, top-1 accuracy, macro-recall, log-loss, multiclass
Brier score and per-class AUROC are reported alongside. Macro-AUROC is preferred because top-1
accuracy is sensitive to class prevalence and forces a single argmax decision. Uncertainty is
expressed as class-stratified, patient-level bootstrap confidence intervals and as the spread across
the 20 outer repeats.

### Module contribution (Shapley)

Module contributions were decomposed by exact Shapley values over all $2^5 = 32$ module subsets. The
characteristic function $v(S)$ is the macro-AUROC of the NNLS late fusion restricted to subset *S*,
with $v(\varnothing) := 0.5$; base models are fixed and only the combiner is refit per subset. For
module *i*,

$$
\phi_i \;=\; \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\,\big[v(S \cup \{i\}) - v(S)\big],
$$

which satisfies the efficiency property $\sum_i \phi_i = v(N) - 0.5$; this identity was checked
numerically in every cohort × tumour-fraction cell. For tumour-fraction-stratified analyses, combiner
weights are fit on all samples and $v(S)$ is evaluated restricted to the bin, so that within each bin
$\sum_i \phi_i$ equals the bin's ensemble macro-AUROC minus 0.5.

## Sequence-bias control

Because local sequence context influences cfDNA cleavage, we tested whether sequence preference alone
could explain the SHAPE signal. A position-specific end-ligation bias model was learned from observed
fragment ends (both ends; the 3′ end reverse-complemented so one matrix applies to either strand)
against a position-matched genomic background, giving log-odds over a 20 bp context (±10 bp) around
the cut. Applied to each site's own reference sequence, this model yields an expected endpoint density
from which a simulated Gini $G^{\text{sim}}_s$ is computed on the identical ±40 bp/5 bp geometry;
the bias-adjusted statistic is

$$
G^{\text{adj}}_s \;=\; G_s - G^{\text{sim}}_s .
$$

Three arms were compared under identical cross-validation: observed Gini, bias-adjusted Gini, and a
**sequence-only** arm using $G^{\text{sim}}$ as the feature. Full derivation, including the both-end
formulation and its phase alignment, is given in Supplementary Methods.

## Cross-assay transfer validation

Models were trained on one version and tested on the other, in both directions, after harmonising the
feature space to the shared target space and shared cancer types. Transfer performance was compared
against the corresponding within-version cross-validation range, and summarised as a stability score
(cross-assay performance relative to within-assay performance).

## External replication in the GRAIL cohort `[PENDING]`

> **This section describes an analysis that has not yet been executed.** As of 2026-07-15, 39 of 198
> raw (analysis1) cfDNA BAMs have finished downloading, skewed toward controls (non-cancer 21/47;
> lung 7/49; prostate 7/54; breast 4/48). The corresponding Results paragraphs are written in
> conditional tense and must not be presented as findings. Text below states the intended procedure.

Independent cfDNA sequencing data from 198 individuals (prostate cancer *n* = 54, lung cancer *n* = 49,
breast cancer *n* = 48, non-cancer controls *n* = 47; one sample per subject) will be analysed with the
same framework. All five modules will be re-extracted *de novo* from the GRAIL sequencing data rather
than transferred, and evaluated by internal cross-validation within the GRAIL cohort; this is therefore
a replication of the **analytical framework**, not of fitted models. The uncollapsed (raw) alignments
are required because the error-corrected/collapsed BAMs do not preserve the fragment endpoints that
SHAPE depends on.

**Sex confounding.** Breast cancer cases are 100% female (48/48) and prostate cancer cases 100% male
(54/54), so these two classes are perfectly separable on sex alone. Any multiclass evaluation must
therefore exclude sex-chromosome-derived signal and/or report sex-stratified performance; otherwise
breast-versus-prostate discrimination is uninterpretable. Lung cancer (32 F/17 M) and controls
(24 F/23 M) are mixed.

## Statistics and reproducibility

Correlations are Spearman's ρ unless stated otherwise; Pearson's *r* is used where a linear
relationship on the stated scale is intended. All cross-validation used fixed random seeds, and the
same outer folds were reused across modules and ablations so that comparisons are paired. No samples
were excluded after the pre-specified cohort rule was applied. Analyses were performed in Python
(NumPy, scikit-learn, XGBoost, pysam) and R (ggplot2/ggpubr).

## Data and code availability

`[VERIFY: accession for the TSO500 cohort, EGA accession EGAD00001005302 for the GRAIL cohort, and a
repository URL/DOI for the analysis code to be inserted.]`
