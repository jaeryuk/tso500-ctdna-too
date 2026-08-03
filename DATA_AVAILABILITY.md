# Data availability

This repository contains **analysis code only**. No patient-level sequencing
data, feature matrices, or credentials are included, and none may be committed
(see `.gitignore`). The data underlying this study are controlled-access and
are governed by the agreements below.

## Clinical TSO500 ctDNA cohort (v1 / v2) and CUP pilot
De-identified clinical cfDNA sequencing generated on the Illumina TruSight
Oncology 500 (TSO500) assay at <INSTITUTION>. These data contain potentially
identifying patient information and are **not publicly available**. They may be
made available to qualified researchers on reasonable request to the
corresponding author and subject to institutional review board (IRB) approval
and a data transfer/use agreement. <ADD IRB / approval number>

## GRAIL / MSK-TECHVAL external replication cohort
Plasma cell-free DNA sequencing from Razavi et al. 2019 (MSK-GRAIL-TECHVAL),
deposited at the European Genome-phenome Archive (EGA) under dataset accession
**EGAD00001005302**. Access is controlled by the relevant EGA Data Access
Committee; apply through https://ega-archive.org . EGA download credentials are
supplied by the user at run time and are never stored in this repository.

## Reference / public resources used
- Human reference genome (build noted per script; b37 / hg19 / hg38 as applicable)
- UniBind transcription-factor binding sites
- CaCTS master transcription-factor database (Sci. Adv. abf6123, Table S6)
- TCGA-ATAC, ENCODE/Cistrome, and hematopoietic ATAC references (accessions in Methods)

## Reproducibility note
All analyses use the **rule-conformant (RC) cohort** definition
(`rc_cohort.py::rc_table`; v1 = 1,093 samples / 10 cancer types,
v2 = 796 samples / 8 cancer types). Feature modules are the finalized
definitions described in `docs/` (SHAPE = per-TFBS endpoint-Gini ⊕
length-entropy, site-filtered; panel-restricted mutation over 519 coding genes;
Helzer-style all-exon depth and exon-1 entropy). Superseded cohorts and feature
variants are not part of this release.
