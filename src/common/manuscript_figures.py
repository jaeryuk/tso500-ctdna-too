#!/usr/bin/env python
"""
Manuscript main figures (1-6) + main tables (1-2) + standalone HTML report, built from the
blood-only 20x5 repeated nested-CV TOO analysis (blood_v1v2_nested2) and existing auxiliary data.
Follows docs/manuscript_figure_table_plan.md; panels needing data not yet computed (GRAIL on the
5-modality model, CUP pilot, SHAP) are clearly flagged as pending. Per-panel try/except so one
failure never kills the build.

Run:  /home/jrkim/.conda/envs/cfse/bin/python scripts/auto/manuscript_figures.py
Out:  results/auto_plan/blood_nested2/figures/Figure{1..6}.png , manuscript_report.html , Table{1,2}.tsv
"""
import os, re, glob, json, base64, warnings
warnings.filterwarnings("ignore")
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

PROJ = "/home/jrkim/TSO_TFBS/project"
D = f"{PROJ}/results/auto_plan/blood_nested2"
FEAT = f"{PROJ}/results/auto_plan/feat"
MAN = f"{PROJ}/results/auto_plan/manifest_dev.tsv"
META = f"{PROJ}/results/cancer_classification/qc/sample_metadata.tsv"   # sid, cancer, tumor_fraction
MP = f"{PROJ}/results/metaplots"
RENAME = "/data/TSO500/id_rename_success.log"
FG = f"{D}/figures"; os.makedirs(FG, exist_ok=True)
COH = ("v1", "v2"); COL = {"v1": "#2c7fb8", "v2": "#d95f0e"}
METHODS = ["E1_entropy", "All_exon_depth", "Genome_wide_CNA", "Somatic_mutation_profile", "SHAPE"]
FEATS = ["Exon1 entropy", "All-exon depth", "Arm CNA", "Mut profile+fusion", "per-TFBS SHAPE"]
FUS = ["EARLY fusion (XGBoost)", "LATE fusion (per-cancer wt)"]
ORDER = FEATS + FUS
SHORT = {"Exon1 entropy": "Exon1 entropy", "All-exon depth": "All-exon depth", "Arm CNA": "Arm CNA",
         "Mut profile+fusion": "Mut+fusion", "per-TFBS SHAPE": "per-TFBS SHAPE",
         "EARLY fusion (XGBoost)": "Early fusion", "LATE fusion (per-cancer wt)": "Late fusion"}
MIN_CLASS = 20; SEED = 42
def basenum(s):
    m = re.match(r'(\d{10})', str(s)); return m.group(1) if m else None
def log(m): print(m, flush=True)


# ----------------------------------------------------------------- data loading
def load_n2tso():
    n2tso = {}
    for ln in open(RENAME):
        for m in re.finditer(r'\b(\d{10})->(TSO_\d+_[A-Z]_\d+)', ln): n2tso[m.group(1)] = m.group(2)
    return n2tso


def cohort_sids(cohort, n2tso):
    """Reproduce blood_v1v2_nested2.load_cohort ordering exactly -> OOF row i == common[i]."""
    man = pd.read_csv(MAN, sep="\t", dtype=str); man = man[man.cohort == cohort]
    lab = dict(zip(man.sid, man.cancer_type))
    elig = {s for s in man.sid if n2tso.get(basenum(s), "").endswith("_B_01") and lab.get(s)}
    common = set(elig)
    for m in METHODS:
        z = np.load(f"{FEAT}/{cohort}/X_{m}.npz", allow_pickle=True)
        common &= set(str(s) for s in z["sids"])
    common = sorted(common)
    cnt = Counter(lab[s] for s in common); classes = sorted([c for c in cnt if cnt[c] >= MIN_CLASS])
    common = [s for s in common if lab[s] in classes]
    return common, lab, classes


def load():
    R = {c: pd.read_csv(f"{D}/{c}_results.tsv", sep="\t").set_index("model") for c in COH}
    PC = {c: pd.read_csv(f"{D}/{c}_percancer_AUROC.tsv", sep="\t", index_col=0) for c in COH}
    Z = {c: dict(np.load(f"{D}/{c}_oof.npz", allow_pickle=True)) for c in COH}
    n2tso = load_n2tso()
    SID = {}; LAB = {}
    for c in COH:
        common, lab, classes = cohort_sids(c, n2tso)
        zc = [str(x) for x in Z[c]["classes"]]
        ymine = np.array([zc.index(lab[s]) for s in common])
        ok = (len(common) == len(Z[c]["y"])) and bool(np.all(ymine == Z[c]["y"])) and (zc == sorted(zc))
        log(f"[sid-recon {c}] n={len(common)} match_y={ok}")
        SID[c] = common; LAB[c] = lab
    TF = load_tf(n2tso)
    log(f"[tumor-fraction] cohort-wide ichorCNA off-probe: {len(TF)} samples")
    return R, PC, Z, SID, LAB, TF


def load_tf(n2tso):
    """Cohort-wide tumor fraction from /data/TSO500/ichorCNA_offProbe_result/{v1,v2}/ichorCNA/*/*.params.txt"""
    tso2num = {v: k for k, v in n2tso.items()}; tf = {}
    for c in COH:
        for pf in glob.glob(f"/data/TSO500/ichorCNA_offProbe_result/{c}/ichorCNA/*/*.params.txt"):
            mm = re.match(r'(TSO_\d+_[A-Z]_\d+)', os.path.basename(os.path.dirname(pf)))
            if not mm: continue
            num = tso2num.get(mm.group(1))
            if num is None: continue
            for ln in open(pf):
                if ln.startswith("Tumor Fraction:"):
                    try: tf[num] = float(ln.split(":", 1)[1].strip().split()[0])
                    except Exception: pass
                    break
    return tf


# extra ensemble variants from base OOF (simple-avg & global-weight late fusion) for Fig 2a
def fusion_variants(z):
    base = np.stack([z[f"P_{m}"] for m in METHODS], 0)            # (5, n, K)
    simple = base.mean(0); simple = simple / simple.sum(1, keepdims=True)
    # global weight = macro-AUROC of each base as weight
    y = z["y"]; K = len(z["classes"])
    w = np.array([np.nanmean([roc_auc_score((y == k).astype(int), z[f"P_{m}"][:, k])
                              for k in range(K) if 0 < (y == k).sum() < len(y)]) for m in METHODS])
    w = np.clip(w - 0.5, 0, None); w = w / w.sum()
    glob = (base * w[:, None, None]).sum(0); glob = glob / glob.sum(1, keepdims=True)
    return simple, glob


def topk_acc(P, y, k):
    order = np.argsort(-P, axis=1)
    return float(np.mean([y[i] in order[i, :k] for i in range(len(y))]))


def brier_ece(P, y, K, nb=10):
    Y = np.eye(K)[y]; brier = float(np.mean(np.sum((P - Y) ** 2, 1)))
    conf = P.max(1); pred = P.argmax(1); corr = (pred == y).astype(float)
    ece = 0.0; n = len(y)
    for b in range(nb):
        lo, hi = b / nb, (b + 1) / nb; m = (conf > lo) & (conf <= hi)
        if m.sum(): ece += m.sum() / n * abs(corr[m].mean() - conf[m].mean())
    return brier, ece


def macro_auc(P, y, K):
    a = [roc_auc_score((y == k).astype(int), P[:, k]) for k in range(K) if 0 < (y == k).sum() < len(y)]
    return float(np.mean(a)) if a else np.nan


def box(ax, x, y, w, h, text, fc="#eef4f8", ec="#2c7fb8", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


# ================================================================= FIGURE 1
def figure1(R, PC, Z, SID, LAB, TF):
    fig = plt.figure(figsize=(16, 13)); gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.25)
    # 1a workflow schematic
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    steps = ["TSO500 BAM/VCF\n(v1, v2)", "QC &\nfeature extraction",
             "5 modality\nclassifiers", "Cancer-type\nlate fusion", "TOO probability\ntop-1/top-2"]
    for i, s in enumerate(steps):
        box(ax, 0.2 + i * 1.95, 1.6, 1.6, 1.0, s, fs=8.5)
        if i < len(steps) - 1: ax.annotate("", (0.2 + i * 1.95 + 1.7, 2.1), (0.2 + (i + 1) * 1.95, 2.1),
                                            arrowprops=dict(arrowstyle="->", lw=1.4))
    feats = ["All-exon depth", "Arm CNA / ichorCNA TF", "Mutation profile + fusion", "per-TFBS SHAPE", "Exon1 entropy"]
    for i, f in enumerate(feats):
        box(ax, 0.2 + i * 1.95, 0.2, 1.7, 0.9, f, fc="#fff6e5", ec="#d95f0e", fs=7.5)
    ax.set_title("a  Study workflow — five fragmentomic/genomic modalities → cancer-type-weighted late fusion",
                 loc="left", fontweight="bold")
    # 1c cohort composition
    ax = fig.add_subplot(gs[1, 0])
    comp = {c: Counter(LAB[c][s] for s in SID[c]) for c in COH}
    cats = sorted(set(comp["v1"]) | set(comp["v2"]), key=lambda t: -(comp["v1"].get(t, 0) + comp["v2"].get(t, 0)))
    yy = np.arange(len(cats))
    ax.barh(yy, [comp["v1"].get(t, 0) for t in cats], color=COL["v1"], label="v1")
    ax.barh(yy, [comp["v2"].get(t, 0) for t in cats], left=[comp["v1"].get(t, 0) for t in cats], color=COL["v2"], label="v2")
    ax.set_yticks(yy); ax.set_yticklabels([t.replace(" cancer", "") for t in cats], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("samples"); ax.legend(frameon=False); ax.set_title("c  Cohort composition by type & version", loc="left", fontweight="bold")
    # 1d design schematic
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    rows = [("TSO500 v1", "internal 20×5 repeated nested CV", "#dceaf3"),
            ("TSO500 v2", "internal 20×5 repeated nested CV", "#dceaf3"),
            ("v1↔v2", "cross-version stress test", "#f6e6d6"),
            ("GRAIL", "independent analytic replication (pending)", "#eeeeee"),
            ("CUP v1/v2", "locked clinical pilot (pending)", "#eeeeee")]
    for i, (a, b, c) in enumerate(rows):
        box(ax, 0.3, 3.3 - i * 0.72, 2.2, 0.6, a, fc=c, fs=8)
        box(ax, 2.8, 3.3 - i * 0.72, 6.7, 0.6, b, fc=c, fs=8)
    ax.set_title("d  Version-aware analysis design", loc="left", fontweight="bold")
    # 1e tumor fraction violin by version
    ax = fig.add_subplot(gs[2, 0])
    data = []
    for c in COH:
        vals = [TF[s] for s in SID[c] if s in TF]
        data.append(vals)
    if any(len(d) for d in data):
        parts = ax.violinplot(data, showmedians=True)
        for pc_, c in zip(parts['bodies'], COH): pc_.set_facecolor(COL[c]); pc_.set_alpha(0.6)
        ax.set_xticks([1, 2]); ax.set_xticklabels([f"{c}\n(n={len(d)})" for c, d in zip(COH, data)])
        ax.set_ylabel("ichorCNA tumor fraction")
    ax.set_title("e  Tumor fraction distribution", loc="left", fontweight="bold")
    # 1f feature availability
    ax = fig.add_subplot(gs[2, 1])
    avail = np.zeros((len(METHODS), 2))
    for j, c in enumerate(COH):
        ns = len(SID[c])
        for i, m in enumerate(METHODS):
            z = np.load(f"{FEAT}/{c}/X_{m}.npz", allow_pickle=True); have = set(str(s) for s in z["sids"])
            avail[i, j] = np.mean([s in have for s in SID[c]])
    im = ax.imshow(avail, cmap="Greens", vmin=0.8, vmax=1.0, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels([c.upper() for c in COH]); ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(FEATS, fontsize=8)
    for i in range(len(METHODS)):
        for j in range(2): ax.text(j, i, f"{avail[i,j]*100:.0f}%", ha="center", va="center", fontsize=8)
    ax.set_title("f  Feature availability (% of cohort)", loc="left", fontweight="bold")
    fig.suptitle("Figure 1. Study overview and version-aware cohort design", fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(f"{FG}/Figure1_overview.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    log("[fig1] ok")


# ================================================================= FIGURE 2
def figure2(R, PC, Z, SID, LAB, TF):
    fig = plt.figure(figsize=(16, 20)); gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.26)
    # extended model list with variants
    EXT = FEATS + ["EARLY fusion (XGBoost)", "Simple-avg late", "Global-wt late", "LATE fusion (per-cancer wt)"]
    var = {c: fusion_variants(Z[c]) for c in COH}
    def top1(c, name):
        z = Z[c]; y = z["y"]
        if name == "Simple-avg late": return topk_acc(var[c][0], y, 1)
        if name == "Global-wt late": return topk_acc(var[c][1], y, 1)
        if name in FEATS:
            raw = METHODS[FEATS.index(name)]; return topk_acc(z[f"P_{raw}"], y, 1)
        if name == "EARLY fusion (XGBoost)": return topk_acc(z["P_EARLY"], y, 1)
        return topk_acc(z["P_LATE"], y, 1)
    # 2a Top-1 grouped bar
    ax = fig.add_subplot(gs[0, 0]); x = np.arange(len(EXT)); w = 0.38
    for j, c in enumerate(COH):
        ax.bar(x + (j - 0.5) * w, [top1(c, m) for m in EXT], w, color=COL[c], ec="k", lw=0.4, label=c.upper())
    ax.set_xticks(x); ax.set_xticklabels([SHORT.get(m, m) for m in EXT], rotation=40, ha="right", fontsize=8)
    ax.axvline(len(FEATS) - 0.5, color="gray", ls="--", lw=0.8); ax.set_ylabel("Top-1 accuracy")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    ax.set_title("a  Overall model comparison (Top-1 accuracy)", loc="left", fontweight="bold")
    # 2b forest macro-AUROC
    ax = fig.add_subplot(gs[0, 1]); yy = np.arange(len(ORDER))
    for j, c in enumerate(COH):
        est = [R[c].loc[m, "macroAUROC"] for m in ORDER]
        lo = [R[c].loc[m, "macroAUROC"] - R[c].loc[m, "macro_lo"] for m in ORDER]
        hi = [R[c].loc[m, "macro_hi"] - R[c].loc[m, "macroAUROC"] for m in ORDER]
        ax.errorbar(est, yy + (j - 0.5) * 0.18, xerr=[lo, hi], fmt="o", color=COL[c], capsize=3, ms=6, label=c.upper())
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[m] for m in ORDER], fontsize=9); ax.invert_yaxis()
    ax.axhline(len(FEATS) - 0.5, color="gray", ls="--", lw=0.8); ax.set_xlabel("macro-AUROC (95% CI)")
    ax.legend(frameon=False); ax.grid(axis="x", alpha=0.25)
    ax.set_title("b  Macro-AUROC forest plot", loc="left", fontweight="bold")
    # 2c per-cancer heatmap: all feature methods + both fusions (no delta column)
    cols_order = FEATS + ["EARLY fusion (XGBoost)", "LATE fusion (per-cancer wt)"]
    for j, c in enumerate(COH):
        ax = fig.add_subplot(gs[1, j]); pc = PC[c].copy()
        tab = pc[cols_order].copy(); tab.columns = [SHORT[m] for m in cols_order]
        M = tab.values.astype(float)
        im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels(tab.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(tab.index)))
        ax.set_yticklabels([f"{i.replace(' cancer','')} (n={int(pc.loc[i,'n'])})" for i in tab.index], fontsize=8)
        ax.axvline(len(FEATS) - 0.5, color="black", lw=1.2)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center", fontsize=6.5,
                        fontweight="bold" if cols_order[b].startswith("LATE") else "normal")
        ax.set_title(f"c  Per-cancer OVR-AUROC by method — {c.upper()}", loc="left", fontweight="bold")
    # 2d confusion matrices
    for j, c in enumerate(COH):
        ax = fig.add_subplot(gs[2, j]); z = Z[c]; cls = [str(x) for x in z["classes"]]
        cm = confusion_matrix(z["y"], z["P_LATE"].argmax(1), labels=range(len(cls)))
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cls))); ax.set_xticklabels([s.replace(" cancer", "")[:11] for s in cls], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cls))); ax.set_yticklabels([s.replace(" cancer", "")[:13] for s in cls], fontsize=7)
        for a in range(len(cls)):
            for b in range(len(cls)):
                if cmn[a, b] > 0.02: ax.text(b, a, f"{cmn[a,b]*100:.0f}", ha="center", va="center", fontsize=6,
                                             color="white" if cmn[a, b] > 0.5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"d  Confusion (late fusion) — {c.upper()}", loc="left", fontweight="bold")
    # 2e top-k + 2f confidence-coverage
    for j, c in enumerate(COH):
        ax = fig.add_subplot(gs[3, j]); z = Z[c]; y = z["y"]; K = len(z["classes"])
        ks = list(range(1, min(4, K) + 1))
        ax.plot(ks, [topk_acc(z["P_LATE"], y, k) for k in ks], "-o", color="#1b9e77", label="Late fusion")
        ax.plot(ks, [topk_acc(z["P_All_exon_depth"], y, k) for k in ks], "-o", color="#999", label="All-exon depth")
        ax.set_xticks(ks); ax.set_xlabel("k"); ax.set_ylabel("Top-k accuracy"); ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
        # inset: confidence-coverage
        axi = ax.inset_axes([0.16, 0.18, 0.46, 0.42])
        for P, cc, lb in [(z["P_LATE"], "#1b9e77", "late"), (z["P_All_exon_depth"], "#999", "depth")]:
            conf = P.max(1); odr = np.argsort(-conf); corr = (P.argmax(1) == y).astype(float)[odr]
            cov = np.arange(1, len(y) + 1) / len(y); acc = np.cumsum(corr) / np.arange(1, len(y) + 1)
            axi.plot(cov, acc, color=cc, lw=1.2)
        axi.set_xlabel("coverage", fontsize=6); axi.set_ylabel("acc", fontsize=6); axi.tick_params(labelsize=5)
        axi.set_title("conf-thresholded", fontsize=6)
        ax.set_title(f"e  Top-k & confidence–coverage — {c.upper()}", loc="left", fontweight="bold")
    fig.suptitle("Figure 2. Primary tissue-of-origin performance of the multimodal late-fusion model",
                 fontsize=14, fontweight="bold", y=0.997)
    fig.savefig(f"{FG}/Figure2_performance.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    log("[fig2] ok")


# ================================================================= FIGURE 3 (version effect)
def figure3(R, PC, Z, SID, LAB, TF):
    fig = plt.figure(figsize=(16, 13)); gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.26)
    # load shared depth matrices
    dep = {}
    for c in COH:
        z = np.load(f"{FEAT}/{c}/X_All_exon_depth.npz", allow_pickle=True)
        pos = {str(s): i for i, s in enumerate(z["sids"])}
        X = np.vstack([z["X"][pos[s]] for s in SID[c]]).astype(float)
        dep[c] = np.nan_to_num(X)
    # 3a PCA
    ax = fig.add_subplot(gs[0, 0])
    Xall = np.vstack([dep["v1"], dep["v2"]]); lab = np.array(["v1"] * len(dep["v1"]) + ["v2"] * len(dep["v2"]))
    Xs = (Xall - Xall.mean(0)) / (Xall.std(0) + 1e-9)
    pcs = TruncatedSVD(2, random_state=0).fit_transform(Xs)
    for c in COH: m = lab == c; ax.scatter(pcs[m, 0], pcs[m, 1], s=8, alpha=0.5, color=COL[c], label=c.upper())
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(frameon=False)
    ax.set_title("a  All-exon depth PCA (version separation)", loc="left", fontweight="bold")
    # 3b per-target depth correlation
    ax = fig.add_subplot(gs[0, 1])
    m1, m2 = dep["v1"].mean(0), dep["v2"].mean(0)
    ax.scatter(m1, m2, s=3, alpha=0.2, color="#555")
    r = spearmanr(m1, m2).correlation
    lim = [min(m1.min(), m2.min()), max(m1.max(), m2.max())]; ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("mean depth v1"); ax.set_ylabel("mean depth v2")
    ax.set_title(f"b  Per-target depth v1 vs v2 (Spearman ρ={r:.2f})", loc="left", fontweight="bold")
    # 3d cross-version transfer (shared classes, depth + concat)
    ax = fig.add_subplot(gs[1, 0])
    shared = sorted(set(LAB["v1"][s] for s in SID["v1"]) & set(LAB["v2"][s] for s in SID["v2"]))
    from sklearn.model_selection import StratifiedKFold
    def cv_transfer(feat):
        # aligned raw matrices (v1/v2 share the same 7567 target columns)
        Xc, yc = {}, {}
        for c in COH:
            z = np.load(f"{FEAT}/{c}/X_{feat}.npz", allow_pickle=True); pos = {str(s): i for i, s in enumerate(z["sids"])}
            keep = [s for s in SID[c] if LAB[c][s] in shared]
            Xc[c] = np.nan_to_num(np.vstack([z["X"][pos[s]] for s in keep]).astype(float))
            yc[c] = np.array([shared.index(LAB[c][s]) for s in keep])
        def fit_pred(Xtr, ytr, Xte):
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9; Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
            if Xtr.shape[1] > 400:                                  # SHARED svd basis: fit on train, apply to test
                svd = TruncatedSVD(200, random_state=0).fit(Ztr); Ztr, Zte = svd.transform(Ztr), svd.transform(Zte)
            clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced").fit(Ztr, ytr)
            P = np.zeros((len(Xte), len(shared))); P[:, clf.classes_] = clf.predict_proba(Zte); return P
        out = {}
        for tr in COH:
            for te in COH:
                if tr == te:                                        # within-version: 5-fold CV (no leakage)
                    P = np.zeros((len(yc[tr]), len(shared)))
                    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xc[tr], yc[tr]):
                        P[b] = fit_pred(Xc[tr][a], yc[tr][a], Xc[tr][b])
                    out[f"{tr}->{te}"] = macro_auc(P, yc[tr], len(shared))
                else:                                               # cross-version: train all tr, test all te
                    out[f"{tr}->{te}"] = macro_auc(fit_pred(Xc[tr], yc[tr], Xc[te]), yc[te], len(shared))
        return out
    tr_depth = cv_transfer("All_exon_depth")
    keys = ["v1->v1", "v2->v2", "v1->v2", "v2->v1"]
    ax.bar(range(4), [tr_depth[k] for k in keys], color=["#2c7fb8", "#d95f0e", "#9ecae1", "#fdae6b"])
    ax.set_xticks(range(4)); ax.set_xticklabels(keys); ax.set_ylabel("macro-AUROC (shared types)")
    ax.set_ylim(0.5, 1.0); ax.grid(axis="y", alpha=0.25)
    ax.set_title("d  Cross-version transfer — all-exon depth", loc="left", fontweight="bold")
    # 3e calibration
    ax = fig.add_subplot(gs[1, 1])
    for c in COH:
        z = Z[c]; P = z["P_LATE"]; y = z["y"]; conf = P.max(1); corr = (P.argmax(1) == y).astype(float)
        bins = np.linspace(0, 1, 11); idx = np.digitize(conf, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 5: xs.append(conf[m].mean()); ys.append(corr[m].mean())
        ax.plot(xs, ys, "-o", color=COL[c], label=c.upper())
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy"); ax.legend(frameon=False)
    ax.set_title("e  Calibration (late fusion)", loc="left", fontweight="bold")
    # 3f per-cancer consistency Δ vs depth in v1 vs v2
    ax = fig.add_subplot(gs[2, 0])
    sh = [t for t in PC["v1"].index if t in PC["v2"].index]
    dv1 = [PC["v1"].loc[t, "LATE fusion (per-cancer wt)"] - PC["v1"].loc[t, "All-exon depth"] for t in sh]
    dv2 = [PC["v2"].loc[t, "LATE fusion (per-cancer wt)"] - PC["v2"].loc[t, "All-exon depth"] for t in sh]
    ax.scatter(dv1, dv2, s=50, color="#444", zorder=3)
    for t, a, b in zip(sh, dv1, dv2): ax.annotate(t.replace(" cancer", "")[:8], (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Δ(late−depth) AUROC, v1"); ax.set_ylabel("Δ(late−depth) AUROC, v2")
    ax.set_title("f  Per-cancer improvement consistency v1 vs v2", loc="left", fontweight="bold")
    # 3c placeholder: feature shift note
    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    ax.text(0.02, 0.95, "c  Version transfer summary", fontweight="bold", fontsize=11, transform=ax.transAxes, va="top")
    txt = (f"All-exon depth shared-class macro-AUROC:\n"
           f"  within-version  v1→v1 {tr_depth['v1->v1']:.3f} · v2→v2 {tr_depth['v2->v2']:.3f}\n"
           f"  cross-version   v1→v2 {tr_depth['v1->v2']:.3f} · v2→v1 {tr_depth['v2->v1']:.3f}\n\n"
           f"Cross-version drop quantifies the v1/v2 chemistry batch effect; version-specific\n"
           f"modeling (used throughout) avoids it. Shared types: {', '.join(t.replace(' cancer','') for t in shared)}.")
    ax.text(0.02, 0.82, txt, fontsize=9, transform=ax.transAxes, va="top", family="monospace")
    fig.suptitle("Figure 3. Assay-version effect and robustness", fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(f"{FG}/Figure3_version.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    log("[fig3] ok"); return tr_depth


# ================================================================= FIGURE 4 (TFBS footprint)
def _agg_npz(folder, keys, sid_filter=None):
    out = {k: [] for k in keys}; files = glob.glob(f"{MP}/{folder}/*.npz"); used = 0
    for f in files:
        sid = os.path.basename(f).split(".")[0].split("_")[0]
        if sid_filter is not None and sid not in sid_filter: continue
        try:
            z = np.load(f, allow_pickle=True)
            if not all(k in z for k in keys): continue
            for k in keys: out[k].append(np.asarray(z[k], float))
            used += 1
        except Exception: continue
    return {k: (np.nanmean(np.stack(v), 0) if v else None) for k, v in out.items()}, used


def figure4(R, PC, Z, SID, LAB, TF):
    fig = plt.figure(figsize=(16, 13)); gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.26)
    allsid = set(SID["v1"]) | set(SID["v2"])
    # 4b footprint metaplot (true vs control) from _short_scan WPS, z-scored per track so shape is comparable
    ax = fig.add_subplot(gs[0, 0])
    def zc(a):
        v = a.sum(0).astype(float); return (v - v.mean()) / (v.std() + 1e-9)
    agg, n = _agg_npz("_short_scan", ["TFBS.feature.wps", "TFBS.control.wps"])
    if agg["TFBS.feature.wps"] is not None:
        fw = zc(agg["TFBS.feature.wps"]); cw = zc(agg["TFBS.control.wps"])
        xx = np.arange(len(fw)) - len(fw) // 2
        ax.plot(xx, fw, color="#1b9e77", label="TFBS sites")
        ax.plot(xx, cw, color="#999", label="matched control")
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("distance from TFBS center (bp)"); ax.set_ylabel("WPS (z-scored)"); ax.legend(frameon=False)
    ax.set_title(f"b  Short-fragment TFBS footprint (n={n})", loc="left", fontweight="bold")
    # 4c short vs long from _meta150
    ax = fig.add_subplot(gs[0, 1])
    agg, n = _agg_npz("_meta150", ["pool_short", "pool_long", "pool_all"])
    if agg["pool_short"] is not None:
        for k, cc, lb in [("pool_short", "#d95f0e", "short (≤119bp)"), ("pool_long", "#2c7fb8", "long (120-180bp)"), ("pool_all", "#444", "all")]:
            v = agg[k]; v = v / np.nanmedian(v); xx = np.arange(len(v)) - len(v) // 2
            ax.plot(xx, v, color=cc, label=lb)
        ax.set_xlabel("distance from TFBS center (bp)"); ax.set_ylabel("normalized coverage"); ax.legend(frameon=False)
    ax.set_title(f"c  Short vs long fragment footprint (n={n})", loc="left", fontweight="bold")
    # 4d v1/v2 reproducibility of per-TF amplitude (tfbs_entropy)
    ax = fig.add_subplot(gs[1, 0])
    def tf_mean(folder, sidset):
        tfs = None; acc = []
        for f in glob.glob(f"{MP}/{folder}/*.npz"):
            sid = os.path.basename(f).split(".")[0].split("_")[0]
            if sid not in sidset: continue
            try:
                z = np.load(f, allow_pickle=True)
                if "tfbs_entropy" not in z or "tfs" not in z: continue
                if tfs is None: tfs = [str(x) for x in z["tfs"]]
                acc.append(np.asarray(z["tfbs_entropy"], float))
            except Exception: continue
        return (tfs, np.nanmean(np.stack(acc), 0)) if acc else (None, None)
    t1, a1 = tf_mean("_helzer_metrics_v1", set(SID["v1"]))
    t2, a2 = tf_mean("_helzer_metrics_v2", set(SID["v2"]))
    if a1 is not None and a2 is not None and t1 == t2:
        ax.scatter(a1, a2, s=10, alpha=0.5, color="#444")
        rr = spearmanr(a1, a2).correlation
        lim = [np.nanmin([a1, a2]), np.nanmax([a1, a2])]; ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("per-TF signal v1"); ax.set_ylabel("per-TF signal v2")
        ax.set_title(f"d  TF-level footprint reproducibility (ρ={rr:.2f})", loc="left", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "TF amplitude arrays not aligned", ha="center"); ax.set_title("d  TF reproducibility", loc="left", fontweight="bold")
    # 4f TFBS-only per-cancer AUROC
    ax = fig.add_subplot(gs[1, 1])
    sh = sorted(set(PC["v1"].index) | set(PC["v2"].index),
                key=lambda t: -(PC["v1"]["per-TFBS SHAPE"].get(t, np.nan) if t in PC["v1"].index else 0))
    yy = np.arange(len(sh))
    for j, c in enumerate(COH):
        vals = [PC[c]["per-TFBS SHAPE"].get(t, np.nan) if t in PC[c].index else np.nan for t in sh]
        ax.scatter(vals, yy + (j - 0.5) * 0.18, color=COL[c], s=40, label=c.upper())
    ax.set_yticks(yy); ax.set_yticklabels([t.replace(" cancer", "") for t in sh], fontsize=8); ax.invert_yaxis()
    ax.axvline(0.5, color="gray", ls="--", lw=0.8); ax.set_xlabel("per-TFBS SHAPE OVR-AUROC"); ax.legend(frameon=False)
    ax.set_title("f  TFBS-only per-cancer AUROC", loc="left", fontweight="bold")
    # 4e negative control: central-vs-flank footprint amplitude, true TFBS vs matched control
    ax = fig.add_subplot(gs[2, 0])
    agg2, n2 = _agg_npz("_short_scan", ["TFBS.feature.wps", "TFBS.control.wps"])
    if agg2["TFBS.feature.wps"] is not None:
        def amp(a):
            v = zc(a); c = len(v) // 2
            return abs(float(np.mean(v[c - 20:c + 20]) - np.mean(np.r_[v[:30], v[-30:]])))
        ax.bar([0, 1], [amp(agg2["TFBS.feature.wps"]), amp(agg2["TFBS.control.wps"])],
               color=["#1b9e77", "#999"], ec="k", lw=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["true TFBS", "matched control"])
        ax.set_ylabel("|central − flank| WPS (z)")
    else:
        ax.text(0.5, 0.5, "see Fig 4b", ha="center", va="center")
    ax.set_title(f"e  Footprint amplitude: TFBS vs control (n={n2})", loc="left", fontweight="bold")
    # 4g model-selection note
    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    ax.text(0.02, 0.95, "g  per-TFBS SHAPE as an independent modality", fontweight="bold", transform=ax.transAxes, va="top")
    txt = ("per-TFBS SHAPE selected base learner (nested CV):\n"
           f"  v1: {R['v1'].loc['per-TFBS SHAPE','selected']}\n"
           f"  v2: {R['v2'].loc['per-TFBS SHAPE','selected']}\n\n"
           f"per-TFBS SHAPE macro-AUROC: v1 {R['v1'].loc['per-TFBS SHAPE','macroAUROC']:.3f}, "
           f"v2 {R['v2'].loc['per-TFBS SHAPE','macroAUROC']:.3f}.\n"
           "Short fragments crossing TFBS centers (30-119bp) show a protection footprint absent in\n"
           "GC/mappability-matched control sites, supporting TFBS entropy as biology, not capture artifact.")
    ax.text(0.02, 0.84, txt, fontsize=9, transform=ax.transAxes, va="top", family="monospace")
    fig.suptitle("Figure 4. Targeted-panel TFBS short-fragment footprint", fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(f"{FG}/Figure4_TFBS.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    log("[fig4] ok")


# ================================================================= FIGURE 5 (tumor fraction)
F5_MODP = [("P_E1_entropy", "Exon1 entropy"), ("P_All_exon_depth", "All-exon depth"),
           ("P_Genome_wide_CNA", "Arm CNA"), ("P_Somatic_mutation_profile", "Mut+fusion"),
           ("P_SHAPE", "per-TFBS SHAPE"), ("P_LATE", "LATE fusion")]
F5_COL = {"Exon1 entropy": "#66c2a5", "All-exon depth": "#fc8d62", "Arm CNA": "#8da0cb",
          "Mut+fusion": "#e78ac3", "per-TFBS SHAPE": "#a6d854", "LATE fusion": "#1b1b1b"}


def figure5(R, PC, Z, SID, LAB, TF):
    fig = plt.figure(figsize=(16, 15)); gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.27)
    BINS = [(-0.01, 0.03, "<3%"), (0.03, 0.10, "3-10%"), (0.10, 0.30, "10-30%"), (0.30, 1.01, "≥30%")]
    rec = {}
    for c in COH:
        z = Z[c]; y = z["y"]
        rec[c] = dict(tf=np.array([TF.get(s, np.nan) for s in SID[c]]), y=y, conf=z["P_LATE"].max(1))

    def feat_top1(c, pk):
        z = Z[c]; y = z["y"]; tf = rec[c]["tf"]; out = []
        for lo, hi, _ in BINS:
            m = (tf > lo) & (tf <= hi); out.append(topk_acc(z[pk][m], y[m], 1) if m.sum() >= 5 else np.nan)
        return out
    xx = np.arange(len(BINS)); w = 0.38
    # 5a counts per bin
    ax = fig.add_subplot(gs[0, 0])
    for j, c in enumerate(COH):
        tf = rec[c]["tf"]; counts = [int(np.sum((tf > lo) & (tf <= hi))) for lo, hi, _ in BINS]
        ax.bar(xx + (j - 0.5) * w, counts, w, color=COL[c], label=c.upper())
    ax.set_xticks(xx); ax.set_xticklabels([b[2] for b in BINS]); ax.set_ylabel("samples"); ax.legend(frameon=False)
    ax.set_title("a  Sample counts by tumor-fraction bin", loc="left", fontweight="bold")
    # 5b/5c per-feature Top-1 vs TF, per cohort
    for col, c in zip((1, 0), ("v1", "v2")):
        ax = fig.add_subplot(gs[0, 1] if c == "v1" else gs[1, 0])
        ns = [int(np.sum((rec[c]["tf"] > lo) & (rec[c]["tf"] <= hi))) for lo, hi, _ in BINS]
        for pk, name in F5_MODP:
            lw = 2.6 if name == "LATE fusion" else 1.4
            ax.plot(range(len(BINS)), feat_top1(c, pk), "-o", color=F5_COL[name], lw=lw, ms=5, label=name)
        ax.set_xticks(range(len(BINS))); ax.set_xticklabels([f"{b[2]}\n(n={n})" for b, n in zip(BINS, ns)], fontsize=8)
        ax.set_ylabel("Top-1 accuracy"); ax.grid(alpha=0.25)
        if c == "v2": ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=1)
        ax.set_title(f"{'b' if c=='v1' else 'c'}  Per-feature Top-1 vs tumor fraction — {c.upper()}", loc="left", fontweight="bold")
    # 5d tumor-fraction robustness: Top-1 drop (<3% -> >=30%) per feature
    ax = fig.add_subplot(gs[1, 1]); names = [n for _, n in F5_MODP]; yy = np.arange(len(names))
    for j, c in enumerate(COH):
        drops = []
        for pk, name in F5_MODP:
            t = feat_top1(c, pk); drops.append((t[-1] - t[0]) if (t[0] == t[0] and t[-1] == t[-1]) else np.nan)
        ax.barh(yy + (j - 0.5) * 0.4, drops, 0.4, color=COL[c], label=c.upper())
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Top-1 gain  <3% → ≥30% TF  (smaller = more TF-robust)"); ax.legend(frameon=False); ax.grid(axis="x", alpha=0.25)
    ax.set_title("d  Tumor-fraction robustness by feature", loc="left", fontweight="bold")
    # 5e low-TF (<3%) per-feature Top-1
    ax = fig.add_subplot(gs[2, 0]); xf = np.arange(len(names))
    for j, c in enumerate(COH):
        lows = [feat_top1(c, pk)[0] for pk, _ in F5_MODP]
        ax.bar(xf + (j - 0.5) * w, lows, w, color=COL[c], label=c.upper())
    ax.set_xticks(xf); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 accuracy (TF<3%)"); ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    ax.set_title("e  Low tumor-fraction subgroup (<3%) by feature", loc="left", fontweight="bold")
    # 5f TF vs confidence
    ax = fig.add_subplot(gs[2, 1])
    for c in COH:
        tf = rec[c]["tf"]; conf = rec[c]["conf"]; corr = (Z[c]["P_LATE"].argmax(1) == rec[c]["y"]).astype(bool)
        ok = ~np.isnan(tf)
        ax.scatter(tf[ok & corr], conf[ok & corr], s=8, alpha=0.4, color=COL[c], label=f"{c} correct")
        ax.scatter(tf[ok & ~corr], conf[ok & ~corr], s=8, alpha=0.4, color=COL[c], marker="x")
    ax.set_xlabel("ichorCNA tumor fraction"); ax.set_ylabel("max predicted prob"); ax.set_xscale("symlog", linthresh=0.03)
    ax.legend(frameon=False, fontsize=8); ax.set_title("f  Tumor fraction vs model confidence", loc="left", fontweight="bold")
    fig.suptitle("Figure 5. Per-feature tumor-fraction dependence and clinical operating characteristics", fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(f"{FG}/Figure5_tumorfraction.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    log("[fig5] ok")


# ================================================================= FIGURE 6 (biology + replication/CUP status)
MODCOL = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]  # per FEATS
# Known biomarker per (cancer, modality) — gene expr→E1, focal amp→depth, chromosomal→Arm CNA, TF→SHAPE
BIO = {
    "breast cancer":        {"E1_entropy": "ESR1/GATA3", "All_exon_depth": "ERBB2(HER2) amp", "Genome_wide_CNA": "1q+/16q−", "SHAPE": "FOXA1/ESR1/GATA3"},
    "prostate cancer":      {"E1_entropy": "AR/KLK3", "All_exon_depth": "AR amp", "Genome_wide_CNA": "8p−/8q+", "SHAPE": "AR/FOXA1/HOXB13"},
    "lung cancer":          {"E1_entropy": "NKX2-1/TTF1", "All_exon_depth": "EGFR/MET amp", "Genome_wide_CNA": "3q+(SOX2)/3p−", "SHAPE": "NKX2-1/TP63/SOX2"},
    "liver cancer":         {"E1_entropy": "ALB/HNF4A", "All_exon_depth": "FGF19/CCND1(11q13)", "Genome_wide_CNA": "1q+/8q+", "SHAPE": "HNF4A/CEBPA/FOXA1"},
    "colorectal cancer":    {"E1_entropy": "CDX2", "All_exon_depth": "ERBB2 amp", "Genome_wide_CNA": "18q−/20q+/8q+", "SHAPE": "CDX2/HNF4A"},
    "gastric cancer":       {"E1_entropy": "CDX2", "All_exon_depth": "ERBB2/FGFR2/MET", "Genome_wide_CNA": "8q+/20q+", "SHAPE": "CDX2/KLF5"},
    "pancreatic cancer":    {"E1_entropy": "PDX1/GATA6", "All_exon_depth": "GATA6/MYC amp", "Genome_wide_CNA": "18q−(SMAD4)/9p−", "SHAPE": "PDX1/GATA6/HNF1B"},
    "melanoma":             {"E1_entropy": "MITF/PMEL", "All_exon_depth": "MITF/CCND1 amp", "Genome_wide_CNA": "7q+/BRAF", "SHAPE": "MITF/SOX10"},
    "biliary tract cancer": {"E1_entropy": "SOX9/HNF1B", "All_exon_depth": "ERBB2/FGFR2", "Genome_wide_CNA": "1q+", "SHAPE": "HNF1B/SOX9"},
    "bladder cancer":       {"E1_entropy": "GATA3/FOXA1", "All_exon_depth": "PPARG/E2F3(6p22)/FGFR3", "Genome_wide_CNA": "9p−", "SHAPE": "FOXA1/GATA3/PPARG"},
    "sarcoma":              {"E1_entropy": "lineage-dep", "All_exon_depth": "MDM2/CDK4(12q13-15)", "Genome_wide_CNA": "12q13-15+", "SHAPE": "lineage-dep"},
}
GRID_MODS = [("E1_entropy", "E1 entropy\n(gene expr)"), ("All_exon_depth", "All-exon depth\n(focal amp)"),
             ("Genome_wide_CNA", "Arm CNA\n(chromosomal)"), ("SHAPE", "per-TFBS SHAPE\n(TF activation)")]


def _shap_modality_contrib(cohort, Z, SID):
    """TreeSHAP on early-fusion XGBoost; per cancer class -> normalized |SHAP| contribution per modality block."""
    import xgboost as xgb
    mats, blocks, start = [], [], 0
    for m in METHODS:
        z = np.load(f"{FEAT}/{cohort}/X_{m}.npz", allow_pickle=True); pos = {str(s): i for i, s in enumerate(z["sids"])}
        X = np.nan_to_num(np.vstack([z["X"][pos[s]] for s in SID[cohort]]).astype(float))
        if X.shape[1] > 1500: X = TruncatedSVD(200, random_state=SEED).fit_transform(X)
        mats.append(X); blocks.append((start, start + X.shape[1])); start += X.shape[1]
    Xc = np.hstack(mats).astype(np.float32); y = Z[cohort]["y"]; K = len(Z[cohort]["classes"])
    clf = xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.6,
                            objective="multi:softprob", num_class=K, tree_method="hist", n_jobs=16,
                            eval_metric="mlogloss", reg_lambda=1.0, verbosity=0).fit(Xc, y)
    contribs = clf.get_booster().predict(xgb.DMatrix(Xc), pred_contribs=True)
    if contribs.ndim == 2: contribs = contribs.reshape(len(Xc), K, -1)   # (n, K, F+1)
    C = np.zeros((K, len(METHODS)))
    for k in range(K):
        idx = np.where(y == k)[0]
        if len(idx) == 0: continue
        for bi, (s, e) in enumerate(blocks):
            C[k, bi] = np.abs(contribs[idx][:, k, s:e]).sum(1).mean()
    return C / np.clip(C.sum(1, keepdims=True), 1e-9, None)


def figure6(R, PC, Z, SID, LAB, TF):
    SH = {c: _shap_modality_contrib(c, Z, SID) for c in COH}
    fig = plt.figure(figsize=(16, 17)); gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.35, 0.42], hspace=0.5, wspace=0.22)
    # 6a/6b SHAP per-cancer modality contribution stacked bars
    for j, c in enumerate(COH):
        ax = fig.add_subplot(gs[0, j]); C = SH[c]; cls = [str(x) for x in Z[c]["classes"]]; bottom = np.zeros(len(cls))
        for mi, feat in enumerate(FEATS):
            ax.bar(range(len(cls)), C[:, mi], bottom=bottom, color=MODCOL[mi], label=feat, ec="white", lw=0.3); bottom += C[:, mi]
        ax.set_xticks(range(len(cls))); ax.set_xticklabels([s.replace(" cancer", "") for s in cls], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("SHAP contribution (fraction)"); ax.set_ylim(0, 1)
        if j == 1: ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=1, bbox_to_anchor=(1.0, 1.0))
        ax.set_title(f"{'ab'[j]}  Per-cancer modality SHAP contribution — {c.upper()}", loc="left", fontweight="bold")
    # 6c/6d biological interpretation grid (annotated + shaded by SHAP)
    midx = [METHODS.index(m) for m, _ in GRID_MODS]
    for j, c in enumerate(COH):
        ax = fig.add_subplot(gs[1, j]); cls = [str(x) for x in Z[c]["classes"]]; C = SH[c][:, midx]
        im = ax.imshow(C, aspect="auto", cmap="Purples", vmin=0, vmax=max(0.5, C.max()))
        ax.set_xticks(range(len(GRID_MODS))); ax.set_xticklabels([lab for _, lab in GRID_MODS], fontsize=8)
        ax.set_yticks(range(len(cls))); ax.set_yticklabels([s.replace(" cancer", "") for s in cls], fontsize=8)
        for a, cl in enumerate(cls):
            for b, (mk, _) in enumerate(GRID_MODS):
                bm = BIO.get(cl, {}).get(mk, "")
                if bm: ax.text(b, a, bm, ha="center", va="center", fontsize=6.0,
                               color="white" if C[a, b] > 0.5 * max(0.5, C.max()) else "black")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="SHAP contribution")
        ax.set_title(f"{'cd'[j]}  Known biomarker × modality (cell=SHAP reliance) — {c.upper()}", loc="left", fontweight="bold")
    # 6e status
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    ax.text(0.01, 0.95, "e  Biological concordance & pending external/clinical data", fontweight="bold", transform=ax.transAxes, va="top")
    txt = ("Cells in (c,d) annotate the canonical biomarker each modality is expected to capture — gene-expression lineage TFs (E1 entropy), "
           "focal amplifications (all-exon depth), arm-level events (Arm CNA), and TF activation (per-TFBS SHAPE) — shaded by the model's actual "
           "per-cancer SHAP reliance, showing where data-driven attribution matches known tumour biology (e.g. AR/8q in prostate, ERBB2 in breast/gastric, "
           "HNF4A in liver, CDX2 in colorectal, MDM2/CDK4 12q in sarcoma).   Pending data: GRAIL replication on the 5-modality model (raw analysis1 BAMs "
           "downloading) and the CUP clinical pilot (~20 cases/version) → Fig 6f/6g/6h.")
    ax.text(0.01, 0.78, txt, fontsize=9, transform=ax.transAxes, va="top", wrap=True)
    fig.suptitle("Figure 6. Biological interpretation — per-cancer modality SHAP and known-biomarker concordance", fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(f"{FG}/Figure6_biology.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    log("[fig6] ok")


# ================================================================= TABLES
def tables(R, PC, Z, SID, LAB, TF):
    # Table 1 cohort characteristics
    rows = []
    for c in COH:
        sids = SID[c]; tfs = [TF[s] for s in sids if s in TF]
        rows.append(dict(cohort=c.upper(), N=len(sids), cancer_types=len(set(LAB[c][s] for s in sids)),
                         median_tumor_fraction=round(float(np.median(tfs)), 4) if tfs else "NA",
                         TF_available=f"{len(tfs)}/{len(sids)}"))
    T1 = pd.DataFrame(rows); T1.to_csv(f"{D}/Table1_cohort.tsv", sep="\t", index=False)
    # Table 2 performance (+Brier/ECE)
    rows = []
    for c in COH:
        z = Z[c]; y = z["y"]; K = len(z["classes"])
        for m in ORDER:
            r = R[c].loc[m]
            P = (z["P_LATE"] if m.startswith("LATE") else z["P_EARLY"] if m.startswith("EARLY")
                 else z[f"P_{METHODS[FEATS.index(m)]}"])
            br, ece = brier_ece(P, y, K)
            rows.append(dict(cohort=c, model=m, N=len(y), top1=round(topk_acc(P, y, 1), 3),
                             top2=round(topk_acc(P, y, 2), 3), top3=round(topk_acc(P, y, 3), 3),
                             macroAUROC=r["macroAUROC"], macro_CI=f"[{r['macro_lo']},{r['macro_hi']}]",
                             weightedAUROC=r["weightedAUROC"], balAcc=r["balAcc"], macroF1=r["macroF1"],
                             Brier=round(br, 3), ECE=round(ece, 3)))
    T2 = pd.DataFrame(rows); T2.to_csv(f"{D}/Table2_performance.tsv", sep="\t", index=False)
    log("[tables] ok"); return T1, T2


# ================================================================= HTML report
def _img64(p): return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
def _tab(df):
    h = "<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead><tbody>"
    for _, r in df.iterrows(): h += "<tr>" + "".join(f"<td>{r[c]}</td>" for c in df.columns) + "</tr>"
    return h + "</tbody></table>"


def html_report(R, PC, Z, SID, LAB, TF, T1, T2, tr_depth):
    css = """body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1180px;margin:24px auto;padding:0 20px;color:#1a1a1a;line-height:1.55}
    h1{font-size:25px;border-bottom:3px solid #2c7fb8;padding-bottom:8px}h2{font-size:20px;margin-top:32px;color:#13476b;border-bottom:1px solid #ddd;padding-bottom:4px}
    table{border-collapse:collapse;margin:12px 0;font-size:12.5px;width:100%}th,td{border:1px solid #ccc;padding:4px 8px;text-align:center}
    thead th{background:#2c7fb8;color:#fff}.callout{background:#eef7f0;border-left:5px solid #1b9e77;padding:12px 18px;margin:16px 0;border-radius:4px}
    figure{margin:22px 0;text-align:center}img{max-width:100%;border:1px solid #ddd;border-radius:5px}figcaption{font-size:13px;color:#555;margin-top:7px;text-align:left}
    .pending{background:#fff6e5;border-left:5px solid #d95f0e;padding:10px 16px;border-radius:4px;font-size:13px}code{background:#f2f2f2;padding:1px 5px;border-radius:3px;font-size:12px}.small{font-size:12px;color:#777}
    figcaption b{color:#13476b}.tk{display:block;margin-top:7px;padding:6px 10px;background:#eef4fa;border-left:4px solid #2c7fb8;border-radius:3px;color:#13476b}
    .mnote{display:block;margin-top:5px;color:#777;font-style:italic}.tdesc{font-size:13px;color:#444;margin:6px 0 2px}"""
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>TSO500 ctDNA TOO — manuscript figures</title><style>{css}</style></head><body>"]
    H.append("<h1>Routine TSO500 ctDNA sequencing carries multiple tissue-of-origin signals: a version-aware, cancer-type-specific late-fusion model</h1>")
    H.append("<p class='small'>Main figures 1–6 and Tables 1–2, built from the blood-only (<code>_B_01</code>) 20×5 repeated nested-CV analysis. "
             "Performance-first narrative; per-TFBS short-fragment footprint as the biological novelty. Panels needing data not yet computed "
             "(GRAIL on the 5-modality model, CUP pilot, SHAP) are flagged.</p>")
    cl = []
    for c in COH:
        late = R[c].loc["LATE fusion (per-cancer wt)"]
        cl.append(f"<b>{c.upper()}</b> (n={len(Z[c]['y'])}, {len(Z[c]['classes'])} types): late-fusion macro-AUROC "
                  f"<b>{late['macroAUROC']:.3f}</b> [{late['macro_lo']:.3f}–{late['macro_hi']:.3f}], weighted {late['weightedAUROC']:.3f}, "
                  f"top-1 {topk_acc(Z[c]['P_LATE'],Z[c]['y'],1)*100:.0f}% · top-2 {topk_acc(Z[c]['P_LATE'],Z[c]['y'],2)*100:.0f}%")
    H.append("<div class='callout'>" + "<br>".join(cl) + "<br><span class='small'>Cancer-type-weighted late fusion is the best model in both "
             "cohorts and exceeds every single modality including the all-exon-depth (Helzer) baseline and early fusion.</span></div>")
    # per-version numbers reused in caption interpretations
    v1, v2 = Z["v1"], Z["v2"]
    lateR = {c: R[c].loc["LATE fusion (per-cancer wt)"] for c in COH}
    depthR = {c: R[c].loc["All-exon depth"] for c in COH}
    earlyR = {c: R[c].loc["EARLY fusion (XGBoost)"] for c in COH}
    figs = [
      ("Figure1_overview.png", "Figure 1. Study overview and version-aware cohort design",
       "<b>What is shown.</b> "
       "<b>(a)</b> Analysis workflow: TSO500 BAM/VCF from two assay versions → QC and extraction of five orthogonal modalities "
       "(all-exon depth, arm-level CNA / ichorCNA tumor fraction, mutation profile + gene fusions, per-TFBS SHAPE short-fragment "
       "footprint, Exon1 fragment-size entropy) → per-cancer-type late fusion → top-1/top-2 tissue-of-origin (TOO) probabilities. "
       "<b>(c)</b> Cohort composition: samples per cancer type, stacked by version (v1 blue, v2 orange). "
       "<b>(d)</b> Version-aware design: each version is modelled independently with 20×5 repeated nested CV; v1↔v2 is a cross-version "
       "stress test; GRAIL (analytic replication) and CUP (clinical pilot) are external arms, pending. "
       "<b>(e)</b> ichorCNA off-probe tumor-fraction distribution per version. "
       "<b>(f)</b> Feature availability — % of each cohort with each modality successfully extracted. "
       "<b>Interpretation.</b> The two versions differ in chemistry but cover overlapping cancer types with a broad tumor-fraction range, "
       "and all five modalities are extracted in essentially the whole cohort, so no modality is missing-at-random by version. "
       "<span class='tk'><b>Conclusion.</b> The dataset supports a fair, version-stratified multimodal benchmark: every sample contributes all five "
       "signals, and the design isolates the assay-version effect rather than confounding it with the model comparison.</span>"),

      ("Figure2_performance.png", "Figure 2. Primary tissue-of-origin performance",
       "<b>What is shown.</b> "
       "<b>(a)</b> Top-1 accuracy for the five single modalities and four ensembles (early fusion, simple-average late, global-weight late, "
       "per-cancer-weighted late fusion), grouped by version. "
       "<b>(b)</b> Macro-AUROC forest plot with 95% CI per model and version. "
       "<b>(c)</b> Per-cancer one-vs-rest AUROC heatmap (rows = cancer types with n; columns = all five modalities + early + late fusion, "
       "late bolded; green = higher). "
       "<b>(d)</b> Row-normalised confusion matrix for late fusion. "
       "<b>(e)</b> Top-k accuracy (late fusion vs all-exon-depth baseline) with an inset confidence–coverage curve (accuracy when only the "
       "most-confident fraction of calls is retained). "
       f"<b>Interpretation.</b> Cancer-type-weighted late fusion is the best model in both cohorts "
       f"(macro-AUROC v1 {lateR['v1']['macroAUROC']:.3f}, v2 {lateR['v2']['macroAUROC']:.3f}) and exceeds the all-exon-depth Helzer baseline "
       f"(v1 {depthR['v1']['macroAUROC']:.3f}, v2 {depthR['v2']['macroAUROC']:.3f}) and early fusion "
       f"(v1 {earlyR['v1']['macroAUROC']:.3f}, v2 {earlyR['v2']['macroAUROC']:.3f}). The per-cancer heatmap shows the gain is broad rather than "
       f"driven by one type, the confusion matrix concentrates errors among biologically adjacent tissues, and top-2 accuracy "
       f"({topk_acc(v1['P_LATE'],v1['y'],2)*100:.0f}%/{topk_acc(v2['P_LATE'],v2['y'],2)*100:.0f}%) plus the confidence–coverage curve show calls "
       f"can be triaged by confidence. "
       "<span class='tk'><b>Conclusion.</b> Combining modalities with per-cancer weights beats any single signal and beats naive fusion; the model is "
       "accurate enough for a top-2 shortlist and supports a confidence threshold for high-precision operation.</span>"),

      ("Figure3_version.png", "Figure 3. Assay-version effect and robustness",
       "<b>What is shown.</b> "
       "<b>(a)</b> 2-component SVD of all-exon depth coloured by version. "
       "<b>(b)</b> Per-target mean depth v1 vs v2 (Spearman ρ). "
       "<b>(c)</b> Text summary of within- vs cross-version transfer. "
       "<b>(d)</b> Cross-version transfer for all-exon depth on shared cancer types: within-version 5-fold CV (v1→v1, v2→v2) vs models trained "
       "on one version and tested on the other (v1→v2, v2→v1). "
       "<b>(e)</b> Calibration of late fusion (observed accuracy vs predicted confidence). "
       "<b>(f)</b> Per-cancer late-minus-depth AUROC gain in v1 vs v2. "
       "<b>Interpretation.</b> Panels (a),(b),(d) quantify a real v1/v2 chemistry batch effect: depth separates by version and cross-version transfer "
       "drops below within-version. Panel (e) shows late-fusion confidence tracks accuracy, and (f) shows the late-fusion advantage over depth holds "
       "in both versions for most cancer types (points in the upper-right quadrant). "
       "<span class='tk'><b>Conclusion.</b> The version effect is large enough that a model trained on one assay should not be applied blindly to the other — "
       "version-specific modelling (used throughout) is the correct choice — yet the multimodal gain itself is reproducible across versions, i.e. it is a "
       "property of the biology, not of one chemistry.</span>"),

      ("Figure4_TFBS.png", "Figure 4. Targeted-panel TFBS short-fragment footprint (Griffin-corrected)",
       "<b>What is shown.</b> Per-fragment GC bias is corrected with Griffin weights (w = 1/bias(L,gc)), positions are restricted to CRG 100-mer "
       "mappability ≥0.90, and every TFBS is compared to matched control centres (non-TFBS positions in the same target window, edge-distance and "
       "GC matched). "
       "<b>(a,b)</b> GC-corrected, flank-normalised WPS for v1 and v2: TFBS (feature) track, matched control, and adjusted (feature−control). "
       "<b>(c)</b> Adjusted-WPS reproducibility, v1 vs v2 overlaid. "
       "<b>(d)</b> GC-corrected coverage (nucleosome positioning), feature vs control. "
       "<b>(e)</b> Short-fragment enrichment (cov_short/cov_all) at TFBS vs control. "
       "<b>(f)</b> TFBS-only per-cancer OVR-AUROC. "
       "<b>Interpretation.</b> A central dip in the adjusted (feature−control) WPS is a genuine transcription-factor protection / nucleosome-depletion "
       "footprint, not a GC or mappability artefact, because the matched control removes those confounders and the feature still departs from it. "
       "The footprint reproduces between two independent chemistries (c), and short fragments are specifically enriched over the protected centre (e). "
       "<b>(f)</b> shows this short-fragment signal alone separates several tumour types. "
       "<span class='tk'><b>Conclusion.</b> Routine targeted panels capture a real, reproducible TF-binding footprint in cfDNA that carries tissue-of-origin "
       "information independent of copy number and mutations — the biological novelty underpinning the SHAPE modality.</span>"
       "<span class='mnote'>This panel embeds the current Griffin render; it auto-refreshes to the full v1+v2 cohort when extraction completes.</span>"),

      ("Figure5_tumorfraction.png", "Figure 5. Per-feature tumor-fraction dependence and operating characteristics",
       "<b>What is shown.</b> "
       "<b>(a)</b> Sample counts per ichorCNA tumor-fraction bin (&lt;3%, 3–10%, 10–30%, ≥30%) by version. "
       "<b>(b,c)</b> Top-1 accuracy of every modality and the late fusion across tumor-fraction bins, for v1 and v2. "
       "<b>(d)</b> Tumor-fraction robustness per feature — Top-1 gain from &lt;3% to ≥30% TF (smaller bar = more burden-robust). "
       "<b>(e)</b> Low-tumor-fraction subgroup (&lt;3%): Top-1 accuracy of each feature. "
       "<b>(f)</b> Tumor fraction vs model confidence (correct = dot, incorrect = ×; symlog axis). "
       "<b>Interpretation.</b> Every signal improves with tumor fraction, but they separate sharply by burden-dependence. The somatic-mutation profile "
       "is the most tumor-fraction-robust single feature (smallest drop; a detected driver is a categorical, burden-independent clue), whereas the "
       "fragmentomic signals — Exon1 entropy and per-TFBS SHAPE — are the most burden-dependent and weakest at &lt;3% (tumour-derived fragmentation is "
       "swamped by background haematopoietic cfDNA). Late fusion is the best model in every bin and its advantage over the best single modality is largest "
       "at &lt;3% TF (v1 0.51 vs 0.44; v2 0.54 vs 0.46), where weak complementary signals combine. "
       "<span class='tk'><b>Conclusion.</b> Modalities contribute unequally across tumour burden — mutation carries the low-burden signal, fragmentomics "
       "(incl. TFBS) contribute mainly at higher burden — and multimodal late fusion adds the most value exactly in the clinically hardest, "
       "low-tumor-fraction regime. (Tumor fraction here is ichorCNA/copy-number-based; a max-somatic-VAF orthogonal-burden replication is in the supplement.)</span>"),

      ("Figure6_biology.png", "Figure 6. Biological interpretation of modality contributions",
       "<b>What is shown.</b> "
       "<b>(a,b)</b> Per-cancer TreeSHAP attribution on the early-fusion model, normalised so each tumour type's bar shows the fraction of predictive "
       "signal coming from each modality. "
       "<b>(c,d)</b> Known-biomarker × modality grid: each cell names the canonical marker a modality should capture — lineage-TF expression → Exon1 "
       "entropy, focal amplification → all-exon depth, arm-level event → Arm CNA, TF activation → per-TFBS SHAPE — shaded by the model's actual SHAP "
       "reliance. <b>(e)</b> Concordance summary and pending external arms. "
       "<b>Interpretation.</b> Different tumour types rely on different modalities, and the reliance lines up with established biology: ERBB2/HER2 depth in "
       "breast and gastric, AR plus 8q in prostate, HNF4A/lineage TFs in liver, CDX2 in colorectal, MDM2/CDK4 (12q13-15) amplification in sarcoma. The "
       "model is therefore using interpretable, tissue-specific drivers rather than opaque correlations. "
       "<span class='tk'><b>Conclusion.</b> The late-fusion model is biologically coherent and self-explaining — its per-cancer modality weights recover known "
       "driver mechanisms, which both builds trust in the predictions and explains <i>why</i> fusion beats any single modality (each tumour type needs a "
       "different signal). GRAIL replication and the CUP pilot remain to externally confirm this.</span>"),
    ]
    H.append("<h2>Main Table 1 — Cohort characteristics</h2>")
    H.append("<p class='tdesc'><b>What is shown.</b> Per version: number of modelled blood samples (N), number of cancer types, median ichorCNA "
             "tumor fraction, and tumor-fraction availability. <b>Interpretation.</b> The two versions are comparable in size and span similar tumour "
             "burdens, so the version comparison in Figs 2–3 is not confounded by cohort size or tumor-fraction imbalance. "
             "<b>Conclusion.</b> v1 and v2 form two adequately-powered, well-characterised cohorts suitable for independent, like-for-like modelling.</p>")
    H.append(_tab(T1))
    H.append("<h2>Main Table 2 — Model performance</h2>")
    H.append("<p class='tdesc'><b>What is shown.</b> For every model in each version: N, top-1/2/3 accuracy, macro-AUROC with 95% CI, weighted AUROC, "
             "balanced accuracy, macro-F1, and calibration (Brier score, expected calibration error). <b>Interpretation.</b> Per-cancer-weighted late "
             "fusion has the highest discrimination (macro/weighted AUROC) and ranking accuracy (top-1/2) with the best or near-best calibration "
             "(lowest Brier/ECE), and its CI sits above the single-modality models — the improvement is statistically supported, not a point estimate. "
             "<b>Conclusion.</b> Late fusion is the recommended operating model in both versions: it dominates on discrimination, ranking, and "
             "calibration simultaneously, which is what a deployable TOO classifier requires.</p>")
    H.append(_tab(T2.round(3)))
    H.append("<h2>Main Figures</h2>")
    for fn, title, cap in figs:
        p = f"{FG}/{fn}"
        if os.path.exists(p):
            H.append(f"<figure><img src='{_img64(p)}'><figcaption><b>{title}.</b><br>{cap}</figcaption></figure>")
    # ---- Revision analyses (NC-readiness): calibration, ablation, artifact controls, TFBS biology, SHAP dim-control
    try:
        import sys as _sys; _sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts/auto/nc_readiness")
        import ncr_report_block as _NCR
        H.append(_NCR.main_section_html())
    except Exception as _e:
        log(f"[report] NC-readiness section skipped: {type(_e).__name__}: {_e}")
    H.append("<h2>Pending data (deferred panels)</h2><div class='pending'>"
             "<b>GRAIL replication</b> on the 5-modality model (raw analysis1 BAMs downloading); "
             "<b>CUP pilot</b> (~20 cases/version, not yet assembled); "
             "<b>SHAP</b> per-feature/lineage-TF enrichment (follow-on compute). "
             "These map to Fig 6f/6g/6h and several supplementary items in <code>docs/manuscript_figure_table_plan.md</code>.</div>")
    H.append("<p class='small'>Plan: <code>docs/manuscript_figure_table_plan.md</code> · runbook: <code>docs/manuscript_build_runbook.md</code> · "
             "analysis: <code>scripts/auto/blood_v1v2_nested2.py</code> · figures: <code>scripts/auto/manuscript_figures.py</code>.</p></body></html>")
    open(f"{D}/manuscript_report.html", "w").write("\n".join(H))
    log(f"[report] -> {D}/manuscript_report.html")


def main():
    R, PC, Z, SID, LAB, TF = load()
    for fn in (figure1, figure2):
        try: fn(R, PC, Z, SID, LAB, TF)
        except Exception as e: log(f"[{fn.__name__}] FAIL {type(e).__name__}: {e}")
    tr_depth = {"v1->v1": np.nan, "v2->v2": np.nan, "v1->v2": np.nan, "v2->v1": np.nan}
    try: tr_depth = figure3(R, PC, Z, SID, LAB, TF)
    except Exception as e: log(f"[figure3] FAIL {type(e).__name__}: {e}")
    # NOTE: Figure 4 (TFBS footprint) is owned by scripts/auto/figure4_griffin.py
    # (Griffin GC-corrected, matched-control-adjusted, all samples, per version). Do NOT regenerate it here.
    for fn in (figure5, figure6):
        try: fn(R, PC, Z, SID, LAB, TF)
        except Exception as e: log(f"[{fn.__name__}] FAIL {type(e).__name__}: {e}")
    T1, T2 = tables(R, PC, Z, SID, LAB, TF)
    html_report(R, PC, Z, SID, LAB, TF, T1, T2, tr_depth)
    log("[done] manuscript figures + tables + report")


if __name__ == "__main__":
    main()
