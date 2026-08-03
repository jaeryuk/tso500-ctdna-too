#!/usr/bin/env python3
"""Lineage-TF contribution-to-TOO vs background, on the rule-conformant cohort restricted to
tumor fraction > 0.03. NOW RUNS BOTH COHORTS v1 AND v2 separately (user directive 2026-07-19), writing a
`cohort` column so the ggpubr renderer can emit one figure per cohort.
Method (per cohort): re-fit enet OVR on the TF>0.03 subset (features re-standardised within the subset);
per-site contribution |beta|*mean|z|; per-TF site-count-matched enrichment z vs the per-cancer population
mean; per-cancer master-vs-background aggregate. Cancers with >= MINPOS samples in the subset only.
Out: results/rule_conformant/_ggpubr_{S,P}_rc_tf03.csv  (both cohorts, `cohort` column)"""
import os, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
PROJ = "/home/jrkim/TSO_TFBS/project"; FEAT = f"{PROJ}/results/auto_plan/feat_rc"
RC = f"{PROJ}/results/rule_conformant"; MAN = f"{RC}/manifest_dev.tsv"; REG = f"{RC}/centers_edge0_regions.tsv"
PADD = f"{PROJ}/results/auto_plan/padgini/npz"; IPAD = 6   # padgini nep col 6 == +-40bp (matches GINI_nep300)
NEP_MIN = int(os.environ.get("DUMBBELL_NEP_MIN", "300"))   # new gini/lenent def: within-sample nep>NEP_MIN gate
TFCUT = 0.03; MINPOS = 10; COHORTS = ("v1", "v2")
# Top-5 (CaCTS-rank order) PANEL-PRESENT master TFs per tumor type, from the CaCTS pan-cancer database
# (Cancer Core Transcription factor Specificity; Science Advances 2021, sciadv.abf6123, Table S6 "candidate
# MTFs"). Higher-ranked CaCTS candidates with NO TFBS in the UniBind panel (e.g. liver CREB3L3/NR1I3/MLXIPL)
# are necessarily excluded. LUAD+LUSC merged -> lung; COAD+READ merged -> colorectal.
MASTER = {
 "prostate cancer": ["NKX3-1","HOXB13","SPDEF","FOXA1","ERG"],
 "lung cancer": ["TP63","SOX2","NFE2L2","XBP1","CEBPD"],
 "colorectal cancer": ["CDX2","HNF4A","KLF5","VDR","ELF3"],
 "breast cancer": ["ESR1","GATA3","FOXA1","XBP1","SPDEF"],
 "gastric cancer": ["HNF4A","KLF5","MECOM","ELF3","EHF"],
 "liver cancer": ["CEBPA","FOXA3","KLF15","HNF4A","HLF"],
 "biliary tract cancer": ["ONECUT2","HNF1B","HNF4A","CEBPB","KLF9"],
 "pancreatic cancer": ["ELF3","BHLHE40","RUNX1","SMAD3","CEBPB"],
 "bladder cancer": ["GATA3","PPARG","ELF3","KLF5","RARG"],
 "melanoma": ["SOX10","MITF","IRF4","TFAP2A","ETV5"],
 "sarcoma": ["MAFB"]}
SHORT = {"prostate cancer":"Prostate","melanoma":"Melanoma","liver cancer":"Liver","colorectal cancer":"Colorectal",
         "gastric cancer":"Gastric","biliary tract cancer":"Biliary","pancreatic cancer":"Pancreatic",
         "bladder cancer":"Bladder","lung cancer":"Lung","sarcoma":"Sarcoma","breast cancer":"Breast"}
CLAB = {"gini":"endpoint-Gini","lenent":"fragment-length-entropy"}


def stars(p): return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<0.05 else "ns"


def load(which, coh, tfcut):
    pre = "gr:" if which=="gini" else "lenhzsp:"
    z = np.load(f"{FEAT}/{coh}/X_SHAPE.npz", allow_pickle=True); X=z["X"]; cols=[str(c) for c in z["cols"]]
    idx=[j for j,c in enumerate(cols) if c.startswith(pre)]; sites=[cols[j][len(pre):] for j in idx]
    sids=[str(s) for s in z["sids"]]
    lab={l.split("\t")[0]:l.split("\t")[2] for l in open(MAN).read().splitlines()[1:] if l.split("\t")[1]==coh}
    if tfcut is not None:
        tftab=f"{PROJ}/results/rcv{coh[-1]}_cna/cna_feature_table.tsv"
        tfser=pd.read_csv(tftab,sep="\t",index_col=0)["tumor_fraction"]; tfser.index=tfser.index.astype(str)
        hitf=set(tfser.index[tfser>tfcut])
        keep=[i for i,s in enumerate(sids) if s in lab and s in hitf]
    else:
        keep=[i for i,s in enumerate(sids) if s in lab]                    # ALL tumor fraction
    y=np.array([lab[sids[i]] for i in keep]); Xk=X[keep].astype(np.float64)[:,idx]
    # NEW gini/lenent def (2026-07-21): percell nep>NEP_MIN site gate matching GINI_nep300/LENENT_nep300.
    # map each gr:/lenhzsp: site -> padgini row via REG index; NaN cells where THAT sample's nep<=NEP_MIN; impute.
    ri={}
    for i,ln in enumerate(open(REG)):
        r=ln.rstrip("\n").split("\t"); ri[f"{r[0]}_{(int(r[1])+int(r[2]))//2}"]=i
    if NEP_MIN>0:
        cvp=np.array([ri[s] for s in sites], np.int64)
        nepM=np.array([(np.load(f"{PADD}/{sids[i]}.padgini.npz",allow_pickle=True)["nep"][:,IPAD].astype(float)[cvp]
                        if os.path.exists(f"{PADD}/{sids[i]}.padgini.npz") else np.full(len(cvp),np.nan)) for i in keep])
        Xk=np.where(nepM>NEP_MIN, Xk, np.nan)
        cm=np.nanmedian(Xk,0); cm=np.where(np.isfinite(cm),cm,0.0); Xk=np.where(np.isfinite(Xk),Xk,cm)  # col-median impute
    else:
        Xk=np.nan_to_num(Xk)
    Xk=(Xk-Xk.mean(0))/(Xk.std(0)+1e-9)                                    # re-standardise within subset
    s2t={}
    for ln in open(REG):
        x=ln.rstrip("\n").split("\t")
        if len(x)>=5: s2t[f"{x[0]}_{(int(x[1])+int(x[2]))//2}"]=x[4]
    return Xk,y,np.array([s2t.get(s,"NA") for s in sites])


def beta_for(Xk,yb): return LogisticRegression(penalty="elasticnet",l1_ratio=0.5,C=0.3,solver="saga",
    class_weight="balanced",max_iter=250,tol=1e-3).fit(Xk,yb).coef_[0]


def rows_for(which, coh, tfcut):
    Xk,y,tfs=load(which,coh,tfcut); from collections import Counter; cnt=Counter(y)
    cancers=[c for c in MASTER if cnt.get(c,0)>=MINPOS]
    betas=dict(zip(cancers,Parallel(n_jobs=5,prefer="processes")(delayed(beta_for)(Xk,(y==c).astype(int)) for c in cancers)))
    Prows=[]; Srows=[]
    for c in cancers:
        b=betas[c]; m=y==c; Cabs=np.abs(b)*np.abs(Xk[m]).mean(0); pm,ps=Cabs.mean(),Cabs.std()+1e-12
        mz=[]
        for t in MASTER[c]:
            sel=np.where(tfs==t)[0]; n=len(sel)
            if n==0: continue
            zt=(Cabs[sel].mean()-pm)/(ps/np.sqrt(n))
            Prows.append(dict(cohort=coh,channel=which,cancer=c,tf=t,z=round(float(zt),3),sig=bool(zt>1.96),
                              ylab=SHORT[c],short=SHORT[c],channel_label=CLAB[which])); mz.append(zt)
        ms=np.isin(tfs,MASTER[c]); npool=int(ms.sum())
        if npool==0 or not mz: continue
        zp=(Cabs[ms].mean()-pm)/(ps/np.sqrt(npool)); p=float(norm.sf(zp))
        Srows.append(dict(cohort=coh,channel=which,cancer=c,master_meanz=round(float(np.mean(mz)),3),rest_meanz=0.0,
                          p=p,wilcox_p=p,stars=stars(p),sig=bool(p<0.05),n_pos=int(cnt[c]),
                          ylab=SHORT[c],short=SHORT[c],channel_label=CLAB[which]))
    return pd.DataFrame(Prows),pd.DataFrame(Srows)


def main():
    P=[]; S=[]
    for coh in COHORTS:
        for w in ("gini","lenent"):
            Pr,Sr=rows_for(w,coh,TFCUT); P.append(Pr); S.append(Sr)
    P=pd.concat(P); S=pd.concat(S)
    P.to_csv(f"{RC}/_ggpubr_P_rc_tf03.csv",index=False); S.to_csv(f"{RC}/_ggpubr_S_rc_tf03.csv",index=False)
    print(S[["cohort","channel","short","n_pos","master_meanz","p","stars"]].round(3).to_string(index=False),flush=True)
    print("TF_CONTRIB_DUMBBELL_RC_TF03_DONE",flush=True)


if __name__=="__main__":
    main()
