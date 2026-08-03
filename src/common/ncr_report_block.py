#!/usr/bin/env python
"""Shared NC-readiness (revision) content for embedding into the MAIN manuscript report and the SUPPLEMENTARY
report. One source of truth for the scorecard + per-pillar What/Interpretation/Conclusion captions and the
figure/table file lists. Import from manuscript_figures.py and manuscript_supplementary.py."""
import os, base64, pandas as pd

NCR = "/home/jrkim/TSO_TFBS/project/results/auto_plan/nc_readiness"
FG, TB = f"{NCR}/figures", f"{NCR}/tables"


def _img64(p): return ("data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()) if os.path.exists(p) else None
def _tabhtml(p, n=40):
    if not os.path.exists(p): return "<i>(missing)</i>"
    return pd.read_csv(p, sep="\t").head(n).to_html(index=False)


SCORECARD = (
 "<table class='ncscore'><tr><th>Critique item</th><th>Status</th><th>Evidence</th></tr>"
 "<tr><td>Probability calibration (ECE, CV-internal)</td><td class='ok'>RESOLVED</td>"
 "<td>Temperature scaling: ECE 0.24→0.042 (v1), 0.20→0.046 (v2); macro-AUROC preserved</td></tr>"
 "<tr><td>TFBS footprint not a probe/edge artifact</td><td class='ok'>STRONG (7/8)</td>"
 "<td>feature≫matched control (p&lt;1e-70); short-specific (opposite long); v1/v2 r=0.92; persists ≥100&nbsp;bp from edge; survives blood exclusion; label-permutation null (p=0.02)</td></tr>"
 "<tr><td>TFBS lineage-TF biology (external CaCTS, FDR, TF&gt;10%)</td><td class='ok'>REPRODUCIBLE (lung)</td>"
 "<td>CaCTS master-TF enrichment BH-FDR-significant for lung in BOTH cohorts (p_BH 0.066/0.087; TP63/SOX2/STAT3); prostate (HOXB13/GRHL2) &amp; breast trending (p_BH≈0.11–0.12)</td></tr>"
 "<tr><td>SHAPE reads cellular composition</td><td class='ok'>STRONG</td>"
 "<td>haematopoietic-TF footprint falls as tumour fraction rises (Pearson r −0.38/−0.57; canonical blood TFs −0.43/−0.60; haem−control −0.41/−0.49) → blood cfDNA diluted by tumour</td></tr>"
 "<tr><td>TFBS incremental performance</td><td class='warn'>SMALL</td>"
 "<td>Δmacro-AUROC remove-TFBS: v1 +0.001 (ns), v2 +0.0045 (CI≈0) → frame as biology, not performance</td></tr>"
 "<tr><td>SHAP dimensionality confound (Fig 6)</td><td class='ok'>CONTROLLED</td>"
 "<td>SHAPE enters early fusion at SVD-200 (=depth, &lt;mutation); dimension-matched late-fusion weight 0.06–0.10 (lowest)</td></tr>"
 "<tr><td>GRAIL replication / CUP pilot</td><td class='bad'>PENDING</td>"
 "<td>GRAIL 396 BAMs present (frag features need extraction); CUP has no labeled cases — remaining gate</td></tr>"
 "</table>")

# (key, title, what, interpretation, conclusion, [figure paths], [(table caption, table path)])
PILLARS = [
 ("P1", "Probability calibration",
  "Four calibrators (temperature, Dirichlet/matrix, isotonic, vs uncalibrated) fit inside a 5-fold CV on the OOF late-/early-fusion probabilities for v1 and v2; metrics = top-label ECE, classwise-ECE, Brier, NLL, macro-AUROC, confidence–coverage.",
  "Uncalibrated late fusion is over-confident (ECE 0.24/0.20); temperature scaling cuts ECE to 0.042 (v1)/0.046 (v2) with macro-AUROC unchanged — better probabilities, same discrimination.",
  "A single CV-internal temperature scaling makes the top-probability trustworthy (ECE&lt;0.10), removing the calibration objection and enabling principled high-confidence calls.",
  [f"{FG}/Fig_P1_reliability.png", f"{FG}/Fig_P1_conf_coverage.png"],
  [("Calibration metrics", f"{TB}/Tab_P1_calibration.tsv")]),
 ("P2", "Modality ablation / TFBS incremental value",
  "Per-cancer NNLS late fusion recomputed on OOF for six modality subsets; bootstrap CI on Δmacro-AUROC for TFBS contrasts; per-cancer ΔAUROC/ΔAUPRC/ΔRecall on adding TFBS; v1/v2 sign-concordance.",
  "Removing TFBS costs only +0.001 macro-AUROC in v1 (ns) and +0.0045 in v2 (CI lower≈0); the gain is real but small and clearest in v2 (biliary, lung, colorectal); v1/v2 sign-concordance 3/6.",
  "TFBS short-fragment signal adds little raw performance over depth+CNA+mutation; its value is independent, interpretable biology (next panels), not a performance gain.",
  [f"{FG}/Fig_P2_ablation.png", f"{FG}/Fig_P2_v1v2_concordance.png"],
  [("Model variants", f"{TB}/Tab_P2_variants.tsv"), ("Per-cancer Δ on adding TFBS", f"{TB}/Tab_P2_percancer_deltaTFBS.tsv")]),
 ("PX", "SHAP dimensionality control",
  "Measured each modality's model-entry dimension and recomputed per-modality TreeSHAP by SUM-over-block (Fig 6 method) vs MEAN-per-feature (dimension-normalised), cross-referenced against the fully dimension-matched late-fusion weight (1 probability/modality).",
  "SHAPE enters early fusion at SVD-200 — same as depth, fewer than mutation — so raw 17k dimensionality never reaches the model; the SHAP-sum still over-credits SHAPE (per-feature share drops to ≈depth), and the dimension-matched late-fusion weight is the lowest (0.055/0.099).",
  "Figure 6's early-fusion SHAP overstates SHAPE via the sum-over-block aggregation, not raw dimensionality. The dimension-matched contribution of TFBS is small — consistent with Pillar 2; present Fig 6 with per-feature-normalised SHAP or late-fusion weights.",
  [], [("SHAP dimensionality control", f"{TB}/Tab_PX_shap_dimcontrol.tsv")]),
 ("P3", "TFBS footprint is not a probe/capture artifact",
  "From Griffin GC/mappability-corrected per-sample footprints (all 1799): feature-vs-matched-control WPS amplitude (paired t) and short-vs-long coverage; v1/v2 reproducibility. At model level (per-site SHAPE, leakage-free CV): macro-AUROC by probe-edge distance (≥25/50/100 bp), with blood/immune sites excluded, under label permutation and TF-label shuffle.",
  "Footprint ≫ matched control (p&lt;1e-70), short-fragment-specific (short coverage dips centrally while long rises, p≈0), v1/v2 reproducible (r=0.92); model signal persists ≥100 bp from any probe edge and after removing 55% immune sites, and collapses under label permutation (0.70 vs 0.50, p=0.02). Only TF-label shuffle is weak (true≈random grouping) — an honest limitation.",
  "Seven of eight artifact controls pass decisively: the short-fragment TFBS footprint is a real, reproducible, capture-independent cfDNA signal — not a GC/mappability/edge/immune/label artifact. This is the central novelty defence and it holds.",
  [f"{FG}/Fig_P3_short_vs_long.png", f"{FG}/Fig_P3_edge_permutation.png"],
  [("Artifact controls", f"{TB}/Tab_P3_controls.tsv")]),
 ("P4", "TFBS attribution vs cancer lineage biology",
  "Clean per-TF short-fraction features rebuilt by aggregating the per-site SHAPE matrix via the site→TF annotation; per-cancer per-TF association = |OVR-AUROC−0.5|; curated lineage-TF enrichment (Mann-Whitney) and v1/v2 reproducibility.",
  "Per-TF attribution is strongly reproducible for well-powered cancers (lung ρ=0.77, colorectal ρ=0.71) and noisy for small classes; curated-lineage enrichment is modest with a clear positive (prostate→HOXB13 top-10, p=0.006).",
  "TFBS attribution carries reproducible, occasionally lineage-specific biology — supporting the mechanistic narrative for the largest cancers — but lineage concordance is partial and class-size-limited; strengthening it (CaCTS/ATAC overlap, larger small-class N) is the clearest next step.",
  [f"{FG}/Fig_P4_lineage_heatmap_v1.png", f"{FG}/Fig_P4_lineage_heatmap_v2.png", f"{FG}/Fig_P4_v1v2_repro.png"],
  [("Lineage enrichment", f"{TB}/Tab_P4_lineage_enrichment.tsv"), ("Per-cancer top TFs", f"{TB}/Tab_P4_percancer_topTF.tsv"),
   ("v1/v2 reproducibility", f"{TB}/Tab_P4_v1v2_repro.tsv")]),
]


def main_section_html():
    """Full 'Revision analyses' section for the MAIN manuscript report (scorecard + key figures + W/I/C)."""
    H = ["<h2>Revision analyses — Nature Communications readiness</h2>",
         "<p class='small'>Added to address reviewer-style critique: calibration, TFBS incremental value, TFBS "
         "artifact controls, TFBS lineage biology, and the early-fusion SHAP dimensionality confound. Full detail + "
         "tables in the supplementary report and <code>results/auto_plan/nc_readiness/nc_readiness_report.html</code>.</p>",
         "<style>.ncscore{border-collapse:collapse;font-size:12px;width:100%}.ncscore td,.ncscore th{border:1px solid #ccc;padding:4px 8px;text-align:left}"
         ".ncscore thead th,.ncscore tr:first-child th{background:#13476b;color:#fff}.ok{color:#1b7a32;font-weight:bold}.warn{color:#b06a00;font-weight:bold}.bad{color:#a11;font-weight:bold}</style>",
         SCORECARD]
    for key, title, what, interp, concl, figs, _ in PILLARS:
        H.append(f"<h3>{key}. {title}</h3>")
        H.append(f"<figcaption><b>What was done.</b> {what} <b>Interpretation.</b> {interp}"
                 f"<span class='tk'><b>Conclusion.</b> {concl}</span></figcaption>")
        for p in figs:
            d = _img64(p)
            if d: H.append(f"<figure><img src='{d}'></figure>")
    return "\n".join(H)


# tumor-fraction per-feature (ichorCNA detail) + orthogonal max-somatic-VAF replication (supplementary only)
TFVAF = [
 ("TF", "Per-feature performance across ichorCNA tumor fraction (detail)",
  "Top-1 accuracy and macro-AUROC of every modality + late fusion across ichorCNA tumor-fraction bins "
  "(&lt;3/3–10/10–30/≥30%), per cohort; heatmap and per-feature burden-robustness (Top-1 gain low→high). "
  "Complements main Figure 5.",
  "All signals improve with tumour fraction, but the somatic-mutation profile is the most burden-robust single "
  "feature (smallest gain) while the fragmentomic signals (Exon1 entropy, per-TFBS SHAPE) are the most "
  "burden-dependent and weakest at &lt;3%; late fusion is best in every bin with the largest advantage at low burden.",
  "Modalities contribute unequally across tumour burden (mutation low-burden, fragmentomics high-burden) and "
  "multimodal fusion adds the most value in the clinically hardest low-tumor-fraction regime.",
  [f"{FG}/Fig_TF_feature_macroAUROC.png", f"{FG}/Fig_TF_feature_heatmap.png"],
  [("Per-feature performance by ichorCNA TF bin", f"{TB}/Tab_TF_feature_performance.tsv"),
   ("Per-feature tumor-fraction robustness", f"{TB}/Tab_TF_feature_robustness.tsv")]),
 ("VAF", "Orthogonal-burden replication using max somatic VAF",
  "Max somatic VAF was derived from the TSO500 CombinedVariantOutput [Small Variants] table (mapped to every "
  "modeled sample via the 'DNA Sample ID' field → 100% coverage in both cohorts). The TSO pipeline does NOT "
  "precompute a max-somatic-VAF/tumor-fraction value, and germline/somatic flags (DRAGEN GermlineStatus) are not "
  "available for the modeled samples; germline was removed by (a) cohort-recurrent high-median-AF positions and "
  "(b) a 0.10 somatic ceiling (germline het rarely &lt;0.10), isolating low-VAF somatic. Per-feature Top-1 across "
  "max-VAF bins + orthogonality scatter vs ichorCNA TF.",
  "Max somatic VAF correlates only moderately with ichorCNA tumor fraction (Spearman 0.30 v1 / 0.46 v2) — a "
  "genuinely independent burden axis — yet the per-feature pattern replicates exactly: Mut+fusion is the most "
  "burden-robust (v1 Top-1 gain 0.003, essentially flat), fragmentomics (entropy, SHAPE) the most burden-dependent, "
  "and late fusion the best in every bin.",
  "The per-feature burden-dependence conclusions are robust to the choice of burden estimate (copy-number "
  "ichorCNA vs mutation max-VAF). Caveat: in a tumour-only panel without matched normal or gnomAD, max somatic VAF "
  "is approximate and low-VAF-isolated (high-burden clonal somatic &gt;10% is capped).",
  [f"{FG}/Fig_VAF_feature_top1.png", f"{FG}/Fig_VAF_vs_ichorTF.png"],
  [("Per-feature performance by max-somatic-VAF bin", f"{TB}/Tab_VAF_feature_performance.tsv"),
   ("Per-feature max-VAF robustness", f"{TB}/Tab_VAF_feature_robustness.tsv")]),
]


BIO_EXTRA = [
 ("P4-CaCTS", "Lineage-TF enrichment — external CaCTS catalog, FDR-corrected, tumour fraction &gt;10%",
  "The hand-curated lineage-TF list was replaced with the external CaCTS master-TF catalog (sciadv.abf6123 "
  "Table S6; TCGA→class mapped), the analysis was restricted to samples with ichorCNA tumour fraction &gt;10% "
  "(reduces haematopoietic dilution), and enrichment of CaCTS master TFs vs background (per-TF |OVR-AUROC−0.5|, "
  "Mann-Whitney) was corrected for multiple testing (BH-FDR + Bonferroni) across all 11 cohort×cancer tests.",
  "Lung cancer is BH-FDR-significant in BOTH cohorts (p_BH 0.066 v1 / 0.087 v2; CaCTS master TFs at the 75–82nd "
  "percentile; TP63/SOX2/STAT3/NR3C1 surfacing), and prostate (HOXB13/GRHL2 in top-10) and breast are just below "
  "the 0.1 FDR threshold (p_BH≈0.11–0.12). This is a marked strengthening over the hand-curated version, where no "
  "cancer survived correction.",
  "With an unbiased external catalog and dilution-controlled (high-tumour-fraction) samples, the TFBS modality "
  "shows reproducible, multiple-testing-significant lineage-TF concordance for the best-powered lineage (lung, both "
  "cohorts), with prostate/breast trending — upgrading the mechanistic claim from 'partial' to 'reproducible for "
  "well-powered cancers'. Small high-TF N still limits rarer types.",
  [f"{FG}/Fig_P4cacts_enrichment.png"],
  [("CaCTS lineage-TF enrichment (FDR-corrected, TF>10%)", f"{TB}/Tab_P4cacts_highTF_enrichment.tsv")]),
 ("HEM", "Haematopoietic-TF footprint vs tumour fraction",
  "Per-sample mean (column-z-scored) SHAPE over haematopoietic TFs (panel blood-enriched TFs + canonical blood "
  "master TFs SPI1/GATA1/TAL1/RUNX1/GFI1B…) correlated with ichorCNA tumour fraction (Pearson + Spearman), with a "
  "non-haematopoietic control and a haematopoietic-minus-control contrast.",
  "The haematopoietic-TF footprint falls steeply as tumour fraction rises (Pearson r −0.38 v1 / −0.57 v2 broad "
  "set; −0.43 / −0.60 for strict canonical blood TFs; all p&lt;1e-40), and the haematopoietic-minus-control "
  "contrast stays strongly negative (−0.41 / −0.49) — the effect is specific to blood TFs, not a global "
  "fragmentation shift.",
  "The SHAPE footprint reads cfDNA cellular composition: as tumour-derived cfDNA increases it replaces "
  "WBC-derived cfDNA and erases the blood-TF protection footprint. This independently validates SHAPE as a genuine "
  "biological readout and is consistent with the blood-exclusion control (Pillar 3f).",
  [f"{FG}/Fig_hemato_tf_vs_tf.png"],
  [("Haematopoietic-TF SHAPE vs tumour fraction (correlations)", f"{TB}/Tab_hemato_tf_vs_tf.tsv")]),
]


def supp_extra_items():
    """(figs, tables) for the per-feature tumor-fraction (ichorCNA) + max-somatic-VAF + CaCTS + haematopoietic blocks."""
    figs, tables = [], []
    for key, title, what, interp, concl, figpaths, tabs in TFVAF + BIO_EXTRA:
        cap = (f"<b>{key}. {title}.</b><br><b>What is shown.</b> {what} <b>Interpretation.</b> {interp}"
               f"<span class='tk'><b>Conclusion.</b> {concl}</span>")
        for i, p in enumerate(figpaths):
            figs.append((cap if i == 0 else f"<b>{key}. {title}</b> (cont.)", p))
        for tcap, tp in tabs:
            tables.append((f"<b>{key}. {title} — {tcap}.</b> {concl}", tp))
    return figs, tables


def supp_items():
    """Return (figs, tables) for the SUPPLEMENTARY report. Each entry: (caption_html, abspath)."""
    figs, tables = [], []
    for key, title, what, interp, concl, figpaths, tabs in PILLARS:
        cap = (f"<b>{key}. {title}.</b><br><b>What is shown.</b> {what} <b>Interpretation.</b> {interp}"
               f"<span class='tk'><b>Conclusion.</b> {concl}</span>")
        for i, p in enumerate(figpaths):
            figs.append((cap if i == 0 else f"<b>{key}. {title}</b> (cont.)", p))
        for tcap, tp in tabs:
            tables.append((f"<b>{key}. {title} — {tcap}.</b> {concl}", tp))
    return figs, tables
