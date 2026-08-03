#!/usr/bin/env python
"""
TSO500 UniBind-TFBS / EE cfDNA fragmentomics pipeline  —  GC + MAPPABILITY corrected.

This is the GC/mappability-corrected reanalysis layer on top of pipeline.py. It keeps the
identical 6 metrics, controls, scoring windows and figures concept, but:

  (A) GRIFFIN-STYLE GC CORRECTION
      Each fragment gets a weight w = 1/bias(L, gc) where bias(L,gc) = observed/expected.
      * observed[L,gc] : (length, reference-GC-count) histogram of the sample's fragments.
      * expected[L,gc] : the genomic background -- for every mappable position in the panel
        analysis universe and every length L, the reference GC of the hypothetical L-mer.
        This is SAMPLE-INDEPENDENT, computed once in `gcmodel`.
      bias is smoothed over gc, clipped, and weights are renormalised so mean weight over
      observed fragments == 1 (total coverage scale preserved). All tracks (coverage,
      short/long coverage, midpoint, WPS) are accumulated with these per-fragment weights.

  (B) MAPPABILITY CORRECTION  (CRG 100-mer, hg19)
      * expected-GC universe counts only positions with mappability >= THR_MAP.
      * callable masks (the per-position normalisation denominator) require within-target
        AND mappability >= THR_MAP, so a mappability trough cannot masquerade as a footprint.
      NB: TSO500 coding-exon targets are ~uniquely mappable (CRG100mer ~1.0), so this filter
      removes only a small fraction of positions; GC is the dominant correction here. The
      excluded fraction is reported in gcmodel logs / QC.

Reference harmonisation:
  BAM / centers / mappability bigWig : 'chr'-prefixed hg19.
  Reference FASTA human_g1k_v37      : NO 'chr' prefix (b37). Autosomes are coordinate-
                                       identical to hg19 (chr1 len 249250621 matches), so we
                                       strip 'chr' for FASTA queries only. Autosomes-only.

Stages:
  gcmodel            build expected GC-by-length model + mappability-filtered callable masks
  sample <id> <bam>  two-pass per BAM: (1) GC-bias model from a target subset, (2) GC-weighted
                     + mappability-aware pooled metaplot tracks  (checkpointed)
  aggregate          control-adjust, score (all 6 metrics), QC, and ALL figures:
                       - {tag}_pooled_metaplots.GCcorr.png   (6-panel, mean over 10 samples)
                       - per_sample/{sid}.GCcorr.png         (6-panel, one per sample)
                       - {tag}_scalar_by_sample.GCcorr.png   (the 2 scalar metrics + others)
                       - qc/gc_bias/{sid}.png                (per-sample GC-bias model QC)
"""
import os, sys
import numpy as np

PROJ  = "/home/jrkim/TSO_TFBS/project"
INTER = f"{PROJ}/intermediate"
CTRLD = f"{INTER}/controls"
GCMD  = f"{INTER}/gcmodel"
RAW   = f"{PROJ}/results/metaplots/_raw_gc"
FASTA = "/home/jrkim/cfdna_se_project/ref/genome/human_g1k_v37.fasta"
MAPBW = "/home/jrkim/cfdna_se_project/ref/mappability/wgEncodeCrgMapabilityAlign100mer.bigWig"
for d in (GCMD, RAW): os.makedirs(d, exist_ok=True)

# ---- parameters (identical analysis geometry to pipeline.py) ----------------
W        = 100
POS      = np.arange(-W, W + 1)            # length 201
MAPQ     = 30
FRAG_MIN, FRAG_MAX = 35, 250
SHORT    = (35, 80)
LONG     = (120, 180)
WPS_HALF = 8
PAD      = FRAG_MAX + W + 5
EPS      = 1e-6
METRIC_KEYS = ["cov_all","cov_short","cov_long","mid_all","mid_short","mid_long","wps_short"]
# ---- correction parameters --------------------------------------------------
THR_MAP        = 0.90          # CRG 100-mer mappability threshold
BIAS_SUBSET    = 1500          # target windows used to estimate per-sample GC bias
GC_SIGMA       = 2.0           # gaussian smoothing of bias over gc (bins)
MIN_EXP_COUNT  = 10.0          # min expected fragment count to trust a (L,gc) bin
WEIGHT_CLIP    = (0.1, 10.0)
SEED           = 42

def log(m): print(m, flush=True)
def contig_of(chrom): return chrom[3:] if chrom.startswith("chr") else chrom

# ---------------------------------------------------------------- center IO
def load_centers(path):
    rows = []
    with open(path) as f:
        for ln in f:
            x = ln.rstrip("\n").split("\t")
            d = dict(chrom=x[0], center=int(x[1]), ts=int(x[2]), te=int(x[3]),
                     tname=x[4], ed=int(x[5]))
            if len(x) > 6: d["name"] = x[6]
            rows.append(d)
    return rows

def tag_groups(tag):
    """Group feature & control centers of a tag by target window (chrom,ts,te)."""
    feats = load_centers(f"{CTRLD}/{tag}.feature.centers.tsv")
    ctrls = load_centers(f"{CTRLD}/{tag}.control.centers.tsv")
    g = {}
    for r in feats: g.setdefault((r["chrom"], r["ts"], r["te"]), {"f": [], "c": []})["f"].append(r)
    for r in ctrls: g.setdefault((r["chrom"], r["ts"], r["te"]), {"f": [], "c": []})["c"].append(r)
    return g

# ---------------------------------------------------------------- reference helpers
def target_seq_map(fa, bw, chrom, ts, te):
    """Return (cs, gcpref, mapbool) for the fetch window [ts-PAD, te+PAD] clipped to contig.
       gcpref: prefix sum of G/C over the window (len R+1). mapbool: mappability>=THR (len R)."""
    contig = contig_of(chrom)
    clen = len(fa[contig])
    cs = max(0, ts - PAD); ce = min(clen, te + PAD)
    seq = fa[contig][cs:ce]
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    gc  = (arr == 71) | (arr == 67)                       # 'G','C' (upper)
    gcpref = np.concatenate([[0], np.cumsum(gc)]).astype(np.int32)
    bwlen = bw.chroms().get(chrom, 0)
    mce = min(ce, bwlen) if bwlen else ce
    mp = np.zeros(ce - cs, dtype=np.float32)
    if mce > cs:
        v = bw.values(chrom, cs, mce, numpy=True)
        mp[:mce - cs] = np.nan_to_num(np.asarray(v, dtype=np.float32))
    return cs, gcpref, (mp >= THR_MAP)

# ================================================================ STAGE: gcmodel
def build_gcmodel():
    import pyBigWig
    from pyfaidx import Fasta
    fa = Fasta(FASTA, as_raw=True, sequence_always_upper=True)
    bw = pyBigWig.open(MAPBW)

    groups = {tag: tag_groups(tag) for tag in ("TFBS", "EE")}
    universe = sorted(set().union(*[set(g.keys()) for g in groups.values()]))
    log(f"[gcmodel] universe target windows: {len(universe)} "
        f"(TFBS {len(groups['TFBS'])}, EE {len(groups['EE'])})")

    expected = np.zeros((FRAG_MAX + 1, FRAG_MAX + 1), dtype=np.float64)  # [L, gc_count]
    callable_ = {tag: {"feature": np.zeros(2*W+1, np.float64),
                       "control": np.zeros(2*W+1, np.float64)} for tag in groups}
    map_pos_tot = 0; map_pos_keep = 0
    Ls = np.arange(FRAG_MIN, FRAG_MAX + 1)

    for i, (chrom, ts, te) in enumerate(universe):
        cs, gcpref, mapbool = target_seq_map(fa, bw, chrom, ts, te)
        R = mapbool.size
        # ---- expected GC background over mappable positions ----
        for L in Ls:
            if R - L + 1 <= 0: break
            gcc = gcpref[L:R+1] - gcpref[0:R-L+1]          # gc count of L-mer starting at p=0..R-L
            valid = mapbool[0:R-L+1]
            if valid.any():
                expected[L, :L+1] += np.bincount(gcc[valid], minlength=L+1)
        map_pos_tot += R; map_pos_keep += int(mapbool.sum())
        # ---- mappability-aware callable for each tag containing this target ----
        for tag, g in groups.items():
            gg = g.get((chrom, ts, te))
            if not gg: continue
            for kind, rows in (("feature", gg["f"]), ("control", gg["c"])):
                out = callable_[tag][kind]
                for r in rows:
                    c = r["center"]
                    gstart = c - W                          # genomic of offset -W
                    loc = gstart - cs                       # local index in window
                    o0 = max(0, -loc); o1 = min(2*W+1, R - loc)
                    if o1 <= o0: continue
                    idx = np.arange(o0, o1)
                    gpos = gstart + idx
                    ok = (gpos >= ts) & (gpos < te) & mapbool[loc + idx]
                    out[idx[ok]] += 1.0
        if (i + 1) % 1000 == 0:
            log(f"[gcmodel] {i+1}/{len(universe)} windows")

    np.savez(f"{GCMD}/expected_gc_by_length.npz", expected=expected,
             Lmin=FRAG_MIN, Lmax=FRAG_MAX, thr_map=THR_MAP)
    for tag in groups:
        np.savez(f"{CTRLD}/{tag}.callable.gcmap.npz",
                 feature=callable_[tag]["feature"], control=callable_[tag]["control"], pos=POS)
    bw.close()
    log(f"[gcmodel] mappable position fraction (THR={THR_MAP}): "
        f"{map_pos_keep/max(map_pos_tot,1):.4f}  ({map_pos_tot-map_pos_keep} of {map_pos_tot} excluded)")
    log("[gcmodel] DONE (expected GC model + mappability callable written).")

# ================================================================ STAGE: sample
def _add_range(diff, a, b, v):
    n = diff.size - 1
    a = max(0, a); b = min(n - 1, b)
    if b >= a:
        diff[a] += v; diff[b + 1] -= v

def build_target_tracks(frags, off, R):
    """frags: list of (fs,fe,L,w). Returns abs-coord arrays, weighted."""
    cov_all = np.zeros(R + 1); cov_sh = np.zeros(R + 1); cov_lo = np.zeros(R + 1)
    wps = np.zeros(R + 1)
    mid_all = np.zeros(R); mid_sh = np.zeros(R); mid_lo = np.zeros(R)
    for fs, fe, L, w in frags:
        a = fs - off; b = fe - off
        short = SHORT[0] <= L <= SHORT[1]; lng = LONG[0] <= L <= LONG[1]
        if 0 <= a < R or 0 <= b <= R or (a < 0 and b > 0):
            aa = max(0, a); bb = min(R, b)
            if bb > aa:
                cov_all[aa] += w; cov_all[bb] -= w
                if short: cov_sh[aa] += w; cov_sh[bb] -= w
                if lng:   cov_lo[aa] += w; cov_lo[bb] -= w
        mid = (fs + fe) // 2 - off
        if 0 <= mid < R:
            mid_all[mid] += w
            if short: mid_sh[mid] += w
            if lng:   mid_lo[mid] += w
        if short:
            _add_range(wps, (fs + WPS_HALF) - off, (fe - 1 - WPS_HALF) - off, +w)
            _add_range(wps, (fs - WPS_HALF) - off, (fs + WPS_HALF) - off, -w)
            _add_range(wps, (fe - 1 - WPS_HALF) - off, (fe - 1 + WPS_HALF) - off, -w)
    return dict(cov_all=np.cumsum(cov_all)[:R], cov_short=np.cumsum(cov_sh)[:R],
                cov_long=np.cumsum(cov_lo)[:R], wps_short=np.cumsum(wps)[:R],
                mid_all=mid_all, mid_short=mid_sh, mid_long=mid_lo)

def accumulate(centers, tracks, off, R, pooled):
    for r in centers:
        c = r["center"]; lo = c - W - off
        if lo < 0 or lo + 2 * W + 1 > R:
            continue
        sl = slice(lo, lo + 2 * W + 1)
        for k in METRIC_KEYS:
            pooled[k] += tracks[k][sl]

def fetch_fragments(bam, chrom, cs, ce, ts, te, gcpref, biasw, qc):
    """Fetch proper read1 fragments; attach reference-GC weight; collect QC."""
    frags = []; gln = gcpref.size - 1
    for read in bam.fetch(chrom, cs, ce):
        if not read.is_read1 or not read.is_proper_pair: continue
        if read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail: continue
        if read.mapping_quality < MAPQ: continue
        tlen = read.template_length
        if tlen <= 0: continue
        L = tlen
        if L < FRAG_MIN or L > FRAG_MAX: continue
        fs = read.reference_start; fe = fs + L
        a = fs - cs; b = fe - cs
        if 0 <= a and b <= gln:
            gc = int(gcpref[b] - gcpref[a]); w = float(biasw[L, gc]) if gc <= L else 1.0
        else:
            w = 1.0
        frags.append((fs, fe, L, w))
        if qc is not None:
            mid = (fs + fe) // 2
            if ts <= mid < te:
                qc["n"] += 1; qc["hist"][min(L, 250)] += 1
                if SHORT[0] <= L <= SHORT[1]: qc["short"] += 1
                if LONG[0] <= L <= LONG[1]:   qc["long"] += 1
    return frags

def tally_observed(bam, chrom, cs, ce, gcpref, observed):
    """Pass-1: tally observed[L,gc] (reference GC) without building tracks."""
    gln = gcpref.size - 1
    for read in bam.fetch(chrom, cs, ce):
        if not read.is_read1 or not read.is_proper_pair: continue
        if read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail: continue
        if read.mapping_quality < MAPQ: continue
        tlen = read.template_length
        if tlen <= 0: continue
        L = tlen
        if L < FRAG_MIN or L > FRAG_MAX: continue
        fs = read.reference_start; fe = fs + L
        a = fs - cs; b = fe - cs
        if 0 <= a and b <= gln:
            gc = int(gcpref[b] - gcpref[a])
            if gc <= L: observed[L, gc] += 1

def build_bias_weights(observed, expected):
    """Griffin-style bias->weight table. weight[L,gc] = 1/(obs/exp), smoothed/clipped/renormalised."""
    from scipy.ndimage import gaussian_filter1d
    weight = np.ones_like(observed, dtype=np.float64)
    for L in range(FRAG_MIN, FRAG_MAX + 1):
        obs = observed[L, :L+1]; exp = expected[L, :L+1]
        nobs = obs.sum()
        if nobs < 50 or exp.sum() == 0: continue
        exp_counts = nobs * (exp / exp.sum())                 # expected fragment count per gc if unbiased
        with np.errstate(divide="ignore", invalid="ignore"):
            bias = np.where(exp_counts >= MIN_EXP_COUNT, obs / exp_counts, np.nan)
            w = 1.0 / bias
        good = np.isfinite(w) & (w > 0)
        if good.sum() < 3:
            continue
        idx = np.arange(w.size)
        w = np.interp(idx, idx[good], w[good])                # fill untrusted bins
        w = gaussian_filter1d(w, GC_SIGMA)
        weight[L, :L+1] = np.clip(w, *WEIGHT_CLIP)
    tot_w = (weight * observed).sum(); tot_n = observed.sum()
    if tot_w > 0: weight *= (tot_n / tot_w)                   # mean weight over observed == 1
    return weight

def run_sample(sid, bampath):
    import pysam, pyBigWig
    from pyfaidx import Fasta
    out = f"{RAW}/{sid}.npz"
    if os.path.exists(out):
        log(f"[sample {sid}] cached, skip"); return
    exp_d = np.load(f"{GCMD}/expected_gc_by_length.npz"); expected = exp_d["expected"]
    fa = Fasta(FASTA, as_raw=True, sequence_always_upper=True)
    bw = pyBigWig.open(MAPBW)
    bam = pysam.AlignmentFile(bampath, "rb")

    TAGS = tuple(t for t in os.environ.get("PGC_TAGS", "TFBS,EE").split(",") if t)
    groups = {tag: tag_groups(tag) for tag in TAGS}
    universe = sorted(set().union(*[set(g.keys()) for g in groups.values()]))

    # ---- PASS 1: estimate per-sample GC bias from a spread-out target subset ----
    stride = max(1, len(universe) // BIAS_SUBSET)
    subset = universe[::stride]
    observed = np.zeros_like(expected)
    for chrom, ts, te in subset:
        cs, gcpref, _ = target_seq_map(fa, bw, chrom, ts, te)
        tally_observed(bam, chrom, cs, cs + gcpref.size - 1, gcpref, observed)
    biasw = build_bias_weights(observed, expected)
    log(f"[sample {sid}] bias: {int(observed.sum())} frags over {len(subset)} windows; "
        f"weight range {biasw[FRAG_MIN:FRAG_MAX+1].min():.2f}-{biasw[FRAG_MIN:FRAG_MAX+1].max():.2f}")

    # ---- PASS 2: GC-weighted, mappability-aware pooled tracks ----
    result = {}
    qc = dict(n=0, short=0, long=0, hist=np.zeros(251, dtype=np.int64))
    seqmap_cache = {}
    for tag in TAGS:
        g = groups[tag]
        pooled_f = {k: np.zeros(2*W+1) for k in METRIC_KEYS}
        pooled_c = {k: np.zeros(2*W+1) for k in METRIC_KEYS}
        nt = len(g); done = 0
        for (chrom, ts, te), gg in g.items():
            key = (chrom, ts, te)
            if key in seqmap_cache:
                cs, gcpref = seqmap_cache[key]
            else:
                cs, gcpref, _ = target_seq_map(fa, bw, chrom, ts, te)
                seqmap_cache[key] = (cs, gcpref)
            off = cs; R = gcpref.size - 1
            frags = fetch_fragments(bam, chrom, cs, cs + R, ts, te, gcpref, biasw, qc)
            tracks = build_target_tracks(frags, off, R)
            if gg["f"]: accumulate(gg["f"], tracks, off, R, pooled_f)
            if gg["c"]: accumulate(gg["c"], tracks, off, R, pooled_c)
            done += 1
            if done % 2000 == 0: log(f"[sample {sid}] {tag}: {done}/{nt} targets")
        for k in METRIC_KEYS:
            result[f"{tag}.feature.{k}"] = pooled_f[k]
            result[f"{tag}.control.{k}"] = pooled_c[k]
        log(f"[sample {sid}] {tag} done ({nt} targets)")
    bam.close(); bw.close()
    result["qc_n"] = np.array([qc["n"]]); result["qc_short"] = np.array([qc["short"]])
    result["qc_long"] = np.array([qc["long"]]); result["qc_hist"] = qc["hist"]
    result["bias_weight"] = biasw; result["gc_observed"] = observed
    np.savez(out, **result)
    log(f"[sample {sid}] SAVED {out}")

# ================================================================ STAGE: aggregate
def savgol(y, win=21, poly=2):
    from scipy.signal import savgol_filter
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < poly + 2: return y
    if win % 2 == 0: win -= 1
    return savgol_filter(y, win, poly)

def slice50(a): return a[W - 50: W + 51]
def norm_profile(raw, callable_):
    c = callable_.astype(float).copy(); c[c == 0] = np.nan
    return raw / c
def score_center_flank(prof, pos, c1=5, f1=20, f2=50):
    center = np.nanmean(prof[(pos >= -c1) & (pos <= c1)])
    flank = np.nanmean(prof[((pos >= -f2) & (pos <= -f1)) | ((pos >= f1) & (pos <= f2))])
    return center - flank
def hf_residual(p): return p - savgol(np.nan_to_num(p))
def fft_spectrum(p):
    y = np.nan_to_num(p); y = y - np.nanmean(y)
    return np.abs(np.fft.rfft(y))
def fft_amp(p):
    amp = fft_spectrum(p)
    return float(np.max(amp[2:])) if amp.size > 2 else 0.0

PROFILE_PANELS = [("shortWPS", "Snyder short-fragment WPS"),
                  ("coverage", "Ulz feature-centered coverage"),
                  ("midpoint", "Griffin midpoint coverage"),
                  ("short_enrichment", "Rao/Snyder short-fragment enrichment"),
                  ("HF_residual", "Ulz high-frequency coverage residual"),
                  ("FFT_spectrum", "Griffin FFT amplitude spectrum")]

def _profiles_for_sample(d, tag, cal_f, cal_c):
    """Return dict of feature/control profiles + scalars for one sample."""
    def prof(kind, metric, cal): return norm_profile(d[f"{tag}.{kind}.{metric}"], cal)
    wps_f, wps_c = prof("feature","wps_short",cal_f), prof("control","wps_short",cal_c)
    cov_f, cov_c = prof("feature","cov_all",cal_f),  prof("control","cov_all",cal_c)
    mid_f, mid_c = prof("feature","mid_all",cal_f),  prof("control","mid_all",cal_c)
    sef = (d[f"{tag}.feature.cov_short"]+EPS)/(d[f"{tag}.feature.cov_all"]+EPS)
    sec = (d[f"{tag}.control.cov_short"]+EPS)/(d[f"{tag}.control.cov_all"]+EPS)
    hf_f, hf_c = hf_residual(cov_f), hf_residual(cov_c)
    ft_f, ft_c = fft_spectrum(mid_f), fft_spectrum(mid_c)
    prof_fc = {"shortWPS": (wps_f, wps_c), "coverage": (cov_f, cov_c),
               "midpoint": (mid_f, mid_c), "short_enrichment": (sef, sec),
               "HF_residual": (hf_f, hf_c), "FFT_spectrum": (ft_f, ft_c)}
    P = POS
    scal = dict(
        adj_shortWPS = score_center_flank(slice50(wps_f),P[W-50:W+51]) - score_center_flank(slice50(wps_c),P[W-50:W+51]),
        adj_coverage = score_center_flank(slice50(cov_f),P[W-50:W+51]) - score_center_flank(slice50(cov_c),P[W-50:W+51]),
        adj_midpoint = float(np.nanmean(slice50(mid_f)[(P[W-50:W+51]>=-30)&(P[W-50:W+51]<=30)]) -
                             np.nanmean(slice50(mid_c)[(P[W-50:W+51]>=-30)&(P[W-50:W+51]<=30)])),
        adj_short_enrich = score_center_flank(slice50(sef),P[W-50:W+51]) - score_center_flank(slice50(sec),P[W-50:W+51]),
        adj_HF_amp = float((np.nanmax(hf_f)-np.nanmin(hf_f)) - (np.nanmax(hf_c)-np.nanmin(hf_c))),
        adj_FFT_amp = float(fft_amp(mid_f) - fft_amp(mid_c)))
    return prof_fc, scal

def _periods(n):
    k = np.arange(fft_spectrum(np.zeros(n)).size)
    per = np.full(k.shape, np.inf); per[1:] = n / k[1:]
    return per

def _draw_panels(fig, axes, prof_fc, title):
    per = _periods(len(POS))
    for ax, (sc, nice) in zip(axes.ravel(), PROFILE_PANELS):
        F, C = prof_fc[sc]
        if sc == "FFT_spectrum":
            m = (per >= 5) & (per <= 220)
            ax.plot(per[m], F[m], label="feature", lw=1.3)
            ax.plot(per[m], C[m], label="control", lw=1.0, alpha=0.8)
            ax.set_xlabel("period (bp)"); ax.set_ylabel("FFT amplitude")
        else:
            ax.plot(POS, F, label="feature", lw=1.3)
            ax.plot(POS, C, label="control", lw=1.0, alpha=0.8)
            ax.plot(POS, F - C, label="adjusted", lw=1.3, color="k")
            ax.axvline(0, color="grey", lw=0.5, ls=":")
            ax.set_xlabel("position rel. to center (bp)")
        ax.set_title(nice, fontsize=9); ax.legend(fontsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

def aggregate():
    import pandas as pd, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    samples = [l.split("\t")[0] for l in open(f"{PROJ}/inputs/sample_manifest.tsv").read().splitlines()[1:]]
    figdir = f"{PROJ}/results/figures"; scoredir = f"{PROJ}/results/scores"; qcdir = f"{PROJ}/results/qc"
    for d in (scoredir, qcdir, f"{qcdir}/gc_bias"): os.makedirs(d, exist_ok=True)

    for tag in ("TFBS", "EE"):
        cal = np.load(f"{CTRLD}/{tag}.callable.gcmap.npz")
        cal_f, cal_c = cal["feature"], cal["control"]
        os.makedirs(f"{figdir}/{tag}/per_sample", exist_ok=True)
        agg = {sc: {"F": [], "C": []} for sc, _ in PROFILE_PANELS}
        score_rows = []
        for sid in samples:
            f = f"{RAW}/{sid}.npz"
            if not os.path.exists(f): continue
            d = np.load(f)
            prof_fc, scal = _profiles_for_sample(d, tag, cal_f, cal_c)
            for sc, _ in PROFILE_PANELS:
                agg[sc]["F"].append(prof_fc[sc][0]); agg[sc]["C"].append(prof_fc[sc][1])
            score_rows.append(dict(sample=sid, tag=tag, **scal))
            # ---- per-sample 6-panel figure ----
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            _draw_panels(fig, axes, prof_fc, f"{tag}  {sid}  (GC+mappability corrected)")
            fig.savefig(f"{figdir}/{tag}/per_sample/{sid}.GCcorr.png", dpi=120); plt.close(fig)
        # ---- scores ----
        df = pd.DataFrame(score_rows)
        df.to_csv(f"{scoredir}/{tag}_pooled_adjusted_scores_by_sample.GCcorr.tsv", sep="\t", index=False)
        # ---- aggregate 6-panel (mean over samples) ----
        mean_fc = {sc: (np.nanmean(np.vstack(agg[sc]["F"]), axis=0),
                        np.nanmean(np.vstack(agg[sc]["C"]), axis=0)) for sc, _ in PROFILE_PANELS}
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        _draw_panels(fig, axes, mean_fc, f"{tag}: GC+mappability-corrected pooled mean over {len(score_rows)} samples")
        fig.savefig(f"{figdir}/{tag}/{tag}_pooled_metaplots.GCcorr.png", dpi=130); plt.close(fig)
        # ---- scalar metrics across samples (incl. the 2 scalar metrics HF & FFT) ----
        cols = ["adj_shortWPS","adj_coverage","adj_midpoint","adj_short_enrich","adj_HF_amp","adj_FFT_amp"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 7))
        x = np.arange(len(df))
        for ax, col in zip(axes.ravel(), cols):
            ax.bar(x, df[col].values, color="steelblue")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(col + ("   ◄ scalar metric" if col in ("adj_HF_amp","adj_FFT_amp") else ""), fontsize=9)
            ax.set_xticks(x); ax.set_xticklabels(df["sample"].values, rotation=90, fontsize=6)
        fig.suptitle(f"{tag}: adjusted scores by sample (GC+mappability corrected)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(f"{figdir}/{tag}/{tag}_scalar_by_sample.GCcorr.png", dpi=130); plt.close(fig)
        log(f"[aggregate] {tag}: scores + aggregate + {len(score_rows)} per-sample + scalar figures written")

    # ---- per-sample GC-bias QC ----
    import matplotlib.pyplot as plt
    for sid in samples:
        f = f"{RAW}/{sid}.npz"
        if not os.path.exists(f): continue
        d = np.load(f)
        if "bias_weight" not in d: continue
        bw_, obs = d["bias_weight"], d["gc_observed"]
        exp = np.load(f"{GCMD}/expected_gc_by_length.npz")["expected"]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for L, col in [(100,"C0"),(150,"C1"),(167,"C2")]:
            o = obs[L,:L+1]; e = exp[L,:L+1]
            gcf = np.arange(L+1)/L
            if o.sum()>0: ax[0].plot(gcf, o/o.sum(), col, lw=1, label=f"obs L={L}")
            if e.sum()>0: ax[0].plot(gcf, e/e.sum(), col, lw=1, ls=":", label=f"exp L={L}")
            ax[1].plot(gcf, bw_[L,:L+1], col, lw=1, label=f"L={L}")
        ax[0].set_title("fragment GC: observed vs expected"); ax[0].set_xlabel("GC fraction"); ax[0].legend(fontsize=6)
        ax[1].set_title("GC correction weight"); ax[1].set_xlabel("GC fraction"); ax[1].axhline(1,color="k",lw=0.5); ax[1].legend(fontsize=6)
        Lmarg_o = obs.sum(axis=1)
        ax[2].plot(np.arange(len(Lmarg_o)), Lmarg_o, lw=1)
        ax[2].set_xlim(0,260); ax[2].set_title("observed fragment-length dist"); ax[2].set_xlabel("length (bp)")
        fig.suptitle(f"GC-bias model QC — {sid}", fontsize=11); fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(f"{qcdir}/gc_bias/{sid}.png", dpi=120); plt.close(fig)

    # ---- fragment QC table + length histogram (corrected run) ----
    qc_rows = []; hists = {}
    for sid in samples:
        f = f"{RAW}/{sid}.npz"
        if not os.path.exists(f): continue
        d = np.load(f); h = d["qc_hist"]; n = int(d["qc_n"][0]); hists[sid] = h
        lens = np.repeat(np.arange(251), h)
        qc_rows.append(dict(sample=sid, on_target_fragments=n,
                            short_35_80=int(d["qc_short"][0]), long_120_180=int(d["qc_long"][0]),
                            short_fraction=round(float(d["qc_short"][0])/max(n,1),4),
                            median_fragment_length=int(np.median(lens)) if n else 0))
    pd.DataFrame(qc_rows).to_csv(f"{qcdir}/fragment_QC_by_sample.GCcorr.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(8,5))
    for sid, h in hists.items(): ax.plot(np.arange(251), h/max(h.sum(),1), lw=0.8, label=sid)
    ax.set_xlabel("fragment length (bp)"); ax.set_ylabel("density"); ax.set_title("Fragment length distribution (corrected run)")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(f"{qcdir}/fragment_length_distribution.GCcorr.png", dpi=130); plt.close(fig)
    log("[aggregate] GC-bias QC + fragment QC written. DONE.")

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gcmodel":     build_gcmodel()
    elif cmd == "sample":    run_sample(sys.argv[2], sys.argv[3])
    elif cmd == "aggregate": aggregate()
    else: sys.exit(f"unknown cmd {cmd}")
