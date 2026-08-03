#!/usr/bin/env python
"""
CN-free / PoN-free TFBS fragmentomic metrics (methods #1 and #5).

Per sample, per retained TFBS center c (full +-ENTWIN window inside target):
  qualifying fragment = read1, proper-pair, primary, not dup/sec/supp/qcfail, MAPQ>=20, 30<=TLEN<=180
  frag_start=reference_start, frag_len=TLEN, frag_mid=fs+len//2, frag_end=fs+len, cut sites = {fs, fe-1}

  #1 short-fraction (CN-free: ratio; PoN-free: vs sample-global):
       n_short = #frags 30-119 with midpoint in [c-SFWIN, c+SFWIN]
       n_long  = #frags 120-180 with midpoint in same window
       (pooled across a TF's sites downstream -> SF_TF; feature = SF_TF - SF_global)

  #5 fragmentation shape (CN-free: functional of a normalized distribution; PoN-free: within-sample):
       ep_entropy = Miller-Madow-corrected Shannon entropy of the endpoint(cut)-position histogram
                    over [c-ENTWIN, c+ENTWIN], normalized by log(#positions)        (footprint flatness)
       ep_gini    = Gini concentration of that endpoint histogram                   (cut-site clustering)
       len_entropy= normalized Shannon entropy of the fragment-length histogram (midpoint in SF window)
       n_ep       = total endpoints in the entropy window (support gate)

Stores per-site float arrays. PoN is NOT used (both metrics are internally referenced).

Modes:
  sample <sid> <bam> <outdir>     extract one BAM -> <outdir>/<sid>.npz
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_gc as P

LEN_LO, LEN_HI = 30, 180
SHORT_HI = 119                  # short = [30,119], long = [120,180]
SFWIN = 30                      # short-fraction / length window half-width (footprint scale)
ENTWIN = 40                     # endpoint-entropy window half-width
EDGE = ENTWIN                   # require full +-ENTWIN window inside target
MAPQ = 20
MARGIN = 220                    # fetch margin: capture frags whose midpoint/endpoints reach the window
NBIN_LEN = LEN_HI - LEN_LO + 1  # 151 length bins
NBIN_EP = 2 * ENTWIN + 1        # 81 single-bp endpoint positions; entropy/gini BINNED offline (1/5/10bp)
def log(m): print(m, flush=True)
def tf_of(name):
    p = (name or "").split("_"); return p[2] if len(p) > 2 else "NA"

def canonical():
    g = P.tag_groups("TFBS"); tkeys = sorted(g.keys()); return g, tkeys

def site_table():
    g, tkeys = canonical(); tf = []; tgt = []
    for ti, k in enumerate(tkeys):
        for r in g[k]["f"]: tf.append(tf_of(r.get("name", ""))); tgt.append(ti)
    return g, tkeys, np.array(tf), np.array(tgt, np.int32)

def shannon_norm(h):
    """Plain normalized Shannon entropy of count vector h, in [0,1] (=H/log(#bins)).
    Finite-sample bias is depth-dependent, so support (n_ep) is stored and residualized at feature stage."""
    n = h.sum()
    if n <= 0: return np.nan
    p = h[h > 0] / n
    return float(-np.sum(p * np.log(p)) / np.log(len(h)))

def gini(h):
    """Gini concentration of a non-negative count vector (0=uniform, ->1-1/n=all mass in one bin)."""
    n = h.size; s = h.sum()
    if s <= 0: return np.nan
    x = np.sort(h.astype(float)); idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / (n * s) - (n + 1) / n)

def run_sample(sid, bampath, outdir):
    import pysam
    os.makedirs(outdir, exist_ok=True)
    out = f"{outdir}/{sid}.npz"
    if os.path.exists(out): log(f"[{sid}] cached"); return
    g, tkeys = canonical()
    Nsites = sum(len(g[k]["f"]) for k in tkeys)
    n_short = np.zeros(Nsites, np.int32); n_long = np.zeros(Nsites, np.int32)
    ep_hist_s = np.zeros((Nsites, NBIN_EP), np.int16)  # SHORT-frag endpoint histogram (TF footprint) -> bin offline
    ep_hist_l = np.zeros((Nsites, NBIN_EP), np.int16)  # LONG-frag endpoint histogram (nucleosome boundary)
    len_ent = np.full(Nsites, np.nan, np.float32)
    n_ep_s = np.zeros(Nsites, np.int32); n_ep_l = np.zeros(Nsites, np.int32)
    valid = np.zeros(Nsites, bool)
    tot_short = 0; tot_long = 0
    bam = pysam.AlignmentFile(bampath, "rb"); gi = 0; done = 0
    for k in tkeys:
        chrom, ts, te = k
        fs_l = []; ln_l = []
        for r in bam.fetch(chrom, max(0, ts - MARGIN), te + MARGIN):
            if not r.is_read1 or not r.is_proper_pair: continue
            if r.is_secondary or r.is_supplementary or r.is_duplicate or r.is_qcfail: continue
            if r.mapping_quality < MAPQ: continue
            L = r.template_length
            if L < LEN_LO or L > LEN_HI: continue
            fs_l.append(r.reference_start); ln_l.append(L)
        nf = len(fs_l)
        pos0 = ts - MARGIN; W = (te + MARGIN) - pos0
        if nf:
            fs = np.asarray(fs_l, np.int64); ln = np.asarray(ln_l, np.int64)
            fe = fs + ln; fmid = fs + ln // 2; sh = ln <= SHORT_HI
            mi = fmid - pos0; mok = (mi >= 0) & (mi < W)
            cs_short = np.cumsum(np.bincount(mi[mok & sh], minlength=W))
            cs_long = np.cumsum(np.bincount(mi[mok & ~sh], minlength=W))
            cstart = fs - pos0; cend = (fe - 1) - pos0      # 5' and 3' cut positions
            def _epcov(mask):
                cc = np.concatenate([cstart[mask], cend[mask]]); ok = (cc >= 0) & (cc < W)
                return np.bincount(cc[ok], minlength=W)
            ep_cov_s = _epcov(sh); ep_cov_l = _epcov(~sh)   # split by fragment size class
        else:
            cs_short = cs_long = ep_cov_s = ep_cov_l = np.zeros(W, np.int64)
            fmid = np.empty(0, np.int64); ln = np.empty(0, np.int64); mi = np.empty(0, np.int64)
        def wsum(cs, a, b):                                   # inclusive [a,b] window sum from cumsum
            a = max(a, 0); b = min(b, W - 1)
            return int(cs[b] - (cs[a - 1] if a > 0 else 0))
        for r in g[k]["f"]:
            c = r["center"]
            if c - EDGE < ts or c + EDGE > te:                # need full entropy window inside target
                gi += 1; continue
            valid[gi] = True
            a_sf = c - SFWIN - pos0; b_sf = c + SFWIN - pos0
            ns = wsum(cs_short, a_sf, b_sf); nl = wsum(cs_long, a_sf, b_sf)
            n_short[gi] = ns; n_long[gi] = nl; tot_short += ns; tot_long += nl
            a_e = c - ENTWIN - pos0
            hs = ep_cov_s[a_e:a_e + NBIN_EP]; hl = ep_cov_l[a_e:a_e + NBIN_EP]
            if hs.size == NBIN_EP:
                ep_hist_s[gi] = np.minimum(hs, 32767); ep_hist_l[gi] = np.minimum(hl, 32767)
                n_ep_s[gi] = int(hs.sum()); n_ep_l[gi] = int(hl.sum())
            if nf:                                            # fragment-length entropy (midpoint in SF window)
                m = (mi >= a_sf) & (mi <= b_sf)
                if m.any():
                    lh = np.bincount((ln[m] - LEN_LO), minlength=NBIN_LEN)
                    len_ent[gi] = shannon_norm(lh)
            gi += 1
        done += 1
        if done % 1000 == 0: log(f"[{sid}] {done}/{len(tkeys)} targets")
    bam.close()
    np.savez_compressed(out, n_short=n_short, n_long=n_long,
             ep_hist_s=ep_hist_s, ep_hist_l=ep_hist_l, n_ep_s=n_ep_s, n_ep_l=n_ep_l,
             len_entropy=len_ent, valid=valid,
             tot_short=np.array([tot_short]), tot_long=np.array([tot_long]), nsites=np.array([Nsites]))
    log(f"[{sid}] SAVED {out}  valid_sites={int(valid.sum())} tot_short={tot_short} tot_long={tot_long} "
        f"SF_glob={tot_short/max(tot_short+tot_long,1):.4f} n_ep_s_med={int(np.median(n_ep_s[valid])) if valid.any() else 0}")

if __name__ == "__main__":
    if sys.argv[1] == "sample": run_sample(sys.argv[2], sys.argv[3], sys.argv[4])
    else: sys.exit(f"unknown cmd {sys.argv[1]}")
