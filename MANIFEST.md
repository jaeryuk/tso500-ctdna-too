# MANIFEST

Every file in this release, by pipeline stage, with its role and the manuscript figure it supports. All analyses use the **rule-conformant (RC) cohort** and the **finalized module definitions only**; superseded cohorts/variants (`keep_t1` MIN_N=8, dropped SHAPE variants, all-gene TMB-trace mutation, the stale rival benchmark dir, dead analyses) and the ~77 download/dedup/EGA-ops scripts are **excluded by design**.

> Shared modules live in `src/common/` and are imported by bare name — run `source env.sh` first to put them on `PYTHONPATH`. The **primary benchmark driver** `rc_benchmark_nested_v2.py` is in `common/` because ~24 scripts import it, but it is the Stage-07 entry point.


## `src/01_cohort/` — Rule-conformant cohort construction  _(→ Fig. 1)_
- `build_rule_conformant_manifest.py` — Build RC cohort manifests (B_01, median coverage >=1000, tumor type >=20/version -> v1=10, v2=8 types; emits keep_t1 column, MIN_N=20)
- `rc_prepare.py` — Resolve RC cohort rows to numeric sids -> manifest_dev.tsv (+ per-cohort ponbench manifests) that every downstream stage joins on

## `src/02_depth/` — Modality 1 — all-exon depth (Helzer-style)  _(→ Fig. 2 / 5)_
- `helzer_metrics_build.py` — Extractor: per-exon all-exon depth (fragments 1-1000 bp, midpoint-assigned, MAPQ>=20, deduplicated) -> _helzer_metrics_rcv{V}

## `src/03_cna/` — Modality 2 — arm-level CNA + tumor fraction (ichorCNA)  _(→ Fig. 2 / 5)_
- `build_cna_broad10mb.py` — Builder: broad 10 Mb arm-level CNA -> X_Genome_wide_CNA (arm cols chr1p..chrXq + tumor_fraction + ploidy; chrX/Y arms excluded downstream)

## `src/04_mutation/` — Modality 3 — panel-restricted (519-gene) mutation  _(→ Fig. 2 / 6)_
- `build_combined_mut.py` — Finalizer: assemble the definitive X_Somatic_mutation_profile (gene-level CVO matrix + curated hotspots)
- `extract_cvo_mutation.py` — Extractor: panel-restricted 519-gene binary mutation (mut_<G> nonsyn + trunc_<G> truncating, genes in >=3 samples) from reportable CVO Small Variants

## `src/05_shape_tfbs/` — Modality 4 — per-TFBS SHAPE (endpoint-Gini + length-entropy)  _(→ Fig. 4)_
- `build_edge0_regions.py` — Build the edge0 TFBS region table (centers_edge0_regions.tsv) consumed by the SHAPE extractors/builder
- `extract_lenent_helzer_span.py` — SHAPE per-site extractor #2: fragment-length entropy over the motif-edge span (insert 20-500 bp, >=1 bp overlap, natural-log)
- `extract_persite_padgini_region.py` — SHAPE per-site extractor #1: endpoint-Gini over the UniBind motif region padded +/-40 bp, 5 bp bins (writes ginipad + per-pad nep support)
- `rc_build_shape_edge0.py` — SHAPE feature-matrix builder: assemble gr:<site> (Gini) + lenhzsp:<site> (length-entropy) channels over the edge0 sites -> X_SHAPE
- `run_lenent_helzer_span.py` — Parallel driver for the per-site length-entropy extractor

## `src/06_exon1_entropy/` — Modality 5 — exon-1 entropy (E1SE, Helzer-exact)  _(→ Fig. 2)_
- `helzer_e1se.py` — Extractor: exon-1 / nearest-exon fragment-length entropy (Helzer-exact; nearest-exon mode -> ~519 genes, 20-500 bp, >=1 bp overlap, MAPQ>=20, nat-log)
- `rebuild_e1_nc.py` — Builder: install the nearest-exon E1SE store into X_E1_entropy

## `src/07_model_latefusion/` — Nested-CV benchmark + cancer-type-specific late fusion  _(→ Fig. 2 / 3)_
- `our_ensemble_best.py` — Best-base + best-combiner six-feature ensemble study (per-feature learner from L2 / elastic-net / XGBoost / HistGB; Stage-2 per-class NNLS-shrink / logistic / XGB fusion)

## `src/08_interpretation/` — Model interpretation (Shapley / LOMO / TF enrichment)  _(→ Fig. 6)_
- `fig_module_contribution.py` — Per-sample / per-cancer NNLS-weighted module-contribution decomposition
- `p4_cacts_highTF.py` — Enrichment against the external CaCTS master-TF catalog (Mann-Whitney master-vs-background + BH-FDR / Bonferroni)
- `per_TF_enrichment.py` — Per-TF informative-site enrichment / specificity on the RC cohort (v1 | v2 side-by-side)
- `rc_shapley.py` — Exact module-level Shapley decomposition of macro-AUROC (2^5 subsets x 20 repeats) from the frozen OOF
- `rc_shapley_interaction.py` — Pairwise Shapley interaction index (module redundancy / synergy)
- `rc_shapley_percancer.py` — Per-cancer exact Shapley + per-cancer LOMO (unique information per class)
- `shap_tf_concordance.py` — TreeSHAP master-TF concordance on the SHAPE channels
- `tf_contribution_dumbbell_rc_all.py` — Lineage-TF contribution vs background, RC cohort, all-TF companion panel
- `tf_contribution_dumbbell_rc_tf03.py` — Lineage-TF contribution-to-TOO vs background, RC cohort, tumor fraction > 0.03 (dumbbell)

## `src/09_grail_replication/` — Independent analytic replication (GRAIL / EGAD00001005302)  _(→ Fig. 6)_
- `grail_baseline_classifier.py` — Train + evaluate the TOO classifier ON GRAIL (leakage-free repeated OOF; 4-class TOO+detection, 3-class TOO, cancer-vs-noncancer)
- `grail_build_features.py` — Master GRAIL feature-matrix builder (Genome_wide_CNA via v2_cna_features, mutation signature, somatic gene profile)
- `grail_cna_features.py` — On-target, PoN-normalized chromosome-arm CNA feature builder
- `grail_crosswalk.py` — Build the EGA metadata crosswalk (EGAF -> EGAN -> phenotype / material / subject)
- `grail_external.py` — TSO500 -> GRAIL transfer endpoint (independent analytic replication; shared-column harmonized 3-class TOO)
- `grail_ichorcna.sh` — Off-probe ichorCNA CNA + tumor fraction (readCounter 1 Mb bins -> runIchorCNA)
- `grail_manifest.py` — Build the labeled GRAIL cfDNA manifest (incl. non-cancer) -> manifest.tsv
- `grail_panel_bed_geneanchored.py` — Build the 508-gene MSK-TechVal panel BED (Razavi Suppl Table 1 -> hg19 exons; TERT promoter special-case, alias fixes)
- `grail_panel_coverage_validate.py` — Per-gene reads/kb coverage validation of the panel BED vs collapsed BAMs
- `grail_panel_exons_depthfilter.py` — Exon-resolution depth intersection of the panel BED (-> depth_covered)
- `grail_raw_offtarget_measure.sh` — Off-target coverage / TLEN feasibility diagnostic on the raw analysis1 BAM
- `grail_readcounter.sh` — 1 Mb readCounter wigs feeding the on-target PoN CNA path
- `phase7_grail.sh` — Orchestrator: manifest -> grail_ichorcna -> grail_build_features -> grail_external

## `src/10_figures/` — Manuscript figure & table generation  _(→ Fig. 2-6)_
- `fig_confusion_ensemble_rc_extract.py` — RC late-fusion confusion matrix (all tumor fraction)
- `fig_confusion_ensemble_rc_tf10_extract.py` — RC late-fusion confusion matrix (tumor fraction > 0.10)
- `fig_shapley_percancer_tfbin.py` — Per-cancer Shapley x tumor-fraction-bin figure
- `p_rc_perf_vs_tfbin.py` — Plot per-tumor-fraction-bin macro-AUROC
- `rc_fig2_data.py` — Rebuild Fig. 2A-D CSVs from benchmark_nested_v2 + RC Shapley (asserts against 0.868 / 0.878)
- `rc_figs_driver.sh` — Idempotent driver that regenerates all RC figures
- `rc_figures_extract.py` — RC per-cohort figure tables (TF gate 0.03; fixed the tumor-fraction source)
- `rc_tfbin_extract.py` — Per-tumor-fraction-bin performance table
- `rc_top1_extract.py` — Top-1 accuracy table (from benchmark_nested_v2)
- `rc_top1_percancer_extract.py` — Per-cancer top-1 accuracy table
- `rerender_rc_figs_on_benchmark.sh` — Re-render the RC figures on the current benchmark output

## `src/common/` — Shared modules & libraries (kept on PYTHONPATH)
- `bench_ctr_vs_pad_cohortA.py` — Shared dependency module (imported by pipeline scripts).
- `build_conc_metrics_bench.py` — Shared dependency module (imported by pipeline scripts).
- `build_lenent_b5_w40.py` — Length-entropy channel helper (resid_logsupport) used by the SHAPE builder
- `chem_separated.py` — Shared dependency module (imported by pipeline scripts).
- `fig_oof_probability.py` — OOF predicted-probability figures (soft confusion, signed margins, per-sample heatmap, calibration)
- `gdd_signatures.py` — 96-channel COSMIC mutation-signature features (used by the GRAIL feature build)
- `manuscript_figures.py` — Shared plotting / style helpers (imported as MF)
- `ncr_common.py` — Shared NC-readiness helpers (load_oof, wnnls, fuse_subset, module/label constants)
- `ncr_report_block.py` — Shared dependency module (imported by pipeline scripts).
- `nonblood_persite.py` — Per-site geometry / robust-z helpers used by the SHAPE builder
- `our_ensemble_v1v2.py` — Shared dependency module (imported by pipeline scripts).
- `p_model_comparison_tuned.py` — Shared dependency module (imported by pipeline scripts).
- `per_cancer_contrib.py` — Shared dependency module (imported by pipeline scripts).
- `pipeline_gc.py` — GC / mappability-corrected fragmentomics feature layer
- `predict_proba_sample.py` — Single-sample OOF probability lookup (also exports rebuild_order)
- `primary.py` — Fusion-math helper library imported by the ensemble scripts
- `rc_assemble.py` — Assembles feat_rc matrices (All_exon_depth builder; copies E1 / SHAPE / CNA from upstream stores)
- `rc_benchmark_nested_v2.py` — *** PRIMARY nested repeated-CV benchmark driver *** (5x20 outer, cross-fit meta-stacker, NNLS-shrink super-learner; per-module + late-fusion macro-AUROC = 0.868 v1 / 0.878 v2; constructs SHAPE_nep300). Also the shared benchmark library imported by ~24 scripts.
- `rc_cohort.py` — Shared RC cohort resolver (rc_table: num_sid, tso_id, cancer)
- `rc_perf_vs_tfbin.py` — Per-tumor-fraction-bin performance (also exports rebuild_order used by the extractors)
- `report_hardening.py` — Shared dependency module (imported by pipeline scripts).
- `v2_cna_features.py` — Arm-level CNA feature builder from ichorCNA output (shared by the GRAIL CNA build; subprocess dep)
- `v2_frag_shape.py` — Shared dependency module (imported by pipeline scripts).

---
_Auto-generated inventory; 70 files total._
