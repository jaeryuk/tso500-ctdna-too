# Tissue-of-origin prediction from routine TSO500 ctDNA sequencing

Analysis code for the study of **multimodal tissue-of-origin (TOO) prediction
from routine clinical TSO500 circulating-tumor-DNA (ctDNA) sequencing**, using a
version-aware, cancer-type-specific **late-fusion** model that integrates five
signal classes and is validated across assay versions and in an independent
external cohort.

> This repository is the *code availability* companion to the manuscript. It
> contains code only — no patient data. See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

## Overview

Five analyzable modalities are extracted from each TSO500 ctDNA sample and
combined by a cancer-type-specific late-fusion classifier:

| # | Modality | What it captures |
|---|----------|------------------|
| 1 | **All-exon depth** | Copy-number / coverage profile across panel exons (Helzer-style) |
| 2 | **Arm-level CNA + tumor fraction** | ichorCNA off-probe copy-number and ctDNA fraction |
| 3 | **Mutation profile** | Gene-level somatic mutation over 519 panel coding genes |
| 4 | **Per-TFBS entropy (SHAPE)** | Short-fragment TF-binding-site footprint: endpoint-Gini ⊕ fragment-length entropy |
| 5 | **Exon-1 entropy (E1SE)** | Promoter-proximal fragment-length entropy |

All analyses use the **rule-conformant (RC) cohort** (`rc_cohort.py::rc_table`):
**v1 = 1,093 samples / 10 cancer types**, **v2 = 796 samples / 8 cancer types**.
The finalized feature definitions (SHAPE site-filtering, panel-restricted
mutation, Helzer-exact depth/E1SE) are the only ones in this release; superseded
cohorts and feature variants are excluded.

## Repository layout

```
tso500-ctdna-too/
├── README.md
├── LICENSE
├── requirements.txt          # Python dependencies
├── DATA_AVAILABILITY.md
├── docs/                     # Methods, supplementary methods, figure/table plan
├── src/
│   ├── common/               # shared modules (cohort table, GC pipeline, plotting helpers)
│   ├── 01_cohort/            # rule-conformant cohort construction (v1 / v2)
│   ├── 02_depth/             # modality 1 — all-exon depth
│   ├── 03_cna/               # modality 2 — ichorCNA arm-level CNA + tumor fraction
│   ├── 04_mutation/          # modality 3 — panel-restricted (519-gene) mutation features
│   ├── 05_shape_tfbs/        # modality 4 — per-TFBS SHAPE (endpoint-Gini ⊕ length-entropy)
│   ├── 06_exon1_entropy/     # modality 5 — Helzer-style exon-1 entropy (E1SE)
│   ├── 07_model_latefusion/  # nested-CV benchmark + cancer-type-specific late fusion
│   ├── 08_interpretation/    # SHAP contributions, lineage/master-TF enrichment
│   ├── 09_grail_replication/ # independent analytic replication (GRAIL / EGAD00001005302)
│   └── 10_figures/           # manuscript figure + table generation
└── MANIFEST.md               # every included script → pipeline stage → manuscript figure
```

## Manuscript figure ↔ code map

| Manuscript | Produced by |
|------------|-------------|
| Fig. 1 — cohort & design | `src/01_cohort/`, `src/10_figures/` |
| Fig. 2 — primary TOO performance | `src/07_model_latefusion/`, `src/10_figures/` |
| Fig. 3 — assay-version effect | `src/07_model_latefusion/`, `src/10_figures/` |
| Fig. 4 — TFBS short-fragment footprint | `src/05_shape_tfbs/`, `src/10_figures/` |
| Fig. 5 — tumor-fraction dependence | `src/03_cna/`, `src/07_model_latefusion/` |
| Fig. 6 — interpretation, GRAIL, CUP | `src/08_interpretation/`, `src/09_grail_replication/` |

(Exact per-figure scripts are listed in [`MANIFEST.md`](MANIFEST.md).)

## Installation

```bash
python -m venv .venv && source .venv/bin/activate    # or conda
pip install -r requirements.txt
```
R figure scripts require: `ggplot2`, `ggpubr`, `data.table`, `ggrepel`,
`scales`, `umap`, `hexbin`.

External tools invoked by some stages: `samtools`, `ichorCNA` (with `hmmcopy`),
and a reference genome FASTA. Paths are configured at the top of each stage's
scripts.

## Usage

Each `src/NN_*/` stage is run in order; feature-extraction stages write matrices
that the modeling stage consumes. See `MANIFEST.md` and the per-stage comments
for inputs/outputs. Raw sequencing data must be obtained separately (see
`DATA_AVAILABILITY.md`).

## Citation

If you use this code, please cite the manuscript (citation to be added on
publication).

## License — Patent Pending, all rights reserved

**This is not open-source software.** The methods in this repository are the
subject of one or more pending or planned patent applications. The code is
provided only for scientific peer review and non-commercial academic
verification of the accompanying manuscript. No patent, commercial, or
redistribution rights are granted. See [LICENSE](LICENSE) for the full terms and
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for data access.

> **Do not make this repository public until a patent application has been
> filed.** Public disclosure (including the manuscript's publication) can bar
> patent rights in most jurisdictions. Coordinate the timing and the license
> terms with your institution's technology transfer office / patent counsel.

Before any public release, fill in the placeholders in `LICENSE` (copyright
holder, licensing contact) and in `DATA_AVAILABILITY.md` (institution, IRB).
