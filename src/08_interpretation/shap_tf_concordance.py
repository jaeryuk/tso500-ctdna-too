#!/usr/bin/env python3
"""SHAP (TreeSHAP) master-TF concordance for SHAPE channels, per-TF resolution (v1).
For each channel (endpoint-Gini, frag-length-entropy) and cancer type c (one-vs-rest): train XGBoost on the
per-TF-aggregated features (252), extract native TreeSHAP contributions (pred_contribs); per-TF importance =
mean_{n in c} |SHAP_{n,t}|. Master-TF concordance: master (lineage) TFs' |SHAP| vs the rest (Mann-Whitney,
master>rest). Dumbbell bg->master per cancer, 2 panels. Nonlinear/interaction-aware complement to enet β·z.
Out: results/rule_conformant/shap_tf_concordance.{png,tsv}"""
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
PROJ="/home/jrkim/TSO_TFBS/project"; FEAT=f"{PROJ}/results/auto_plan/feat_rc"
RC=f"{PROJ}/results/rule_conformant"; MAN=f"{RC}/manifest_dev.tsv"; REG=f"{RC}/centers_edge0_regions.tsv"
OUT=f"{RC}/shap_tf_concordance.png"; TSV=f"{RC}/shap_tf_concordance.tsv"
MASTER={"prostate cancer":["AR","FOXA1","HOXB13","NKX3-1"],"melanoma":["MITF","SOX10","TFAP2A"],
 "liver cancer":["HNF4A","HNF1A","FOXA3","CEBPA","ONECUT1"],"colorectal cancer":["CDX2","HNF4A","TCF7L2","KLF5"],
 "gastric cancer":["GATA4","GATA6","CDX2","HNF4A","FOXA2"],"biliary tract cancer":["HNF1B","HNF4A","HNF1A","ONECUT1"],
 "pancreatic cancer":["PDX1","HNF1B","GATA6","NR5A2"],"bladder cancer":["GATA3","PPARG","TP63","KLF5"],
 "lung cancer":["SOX2","FOXA1","FOXA2","TP63"],"sarcoma":["MYOD1","TWIST1","SNAI2"]}
SHORT={k:k.split()[0].capitalize() for k in MASTER}; ROWS=list(MASTER)


def stars(p): return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<0.05 else "ns"


def load_pertf(which):
    pre="gr:" if which=="gini" else "lenhzsp:"
    z=np.load(f"{FEAT}/v1/X_SHAPE.npz",allow_pickle=True); X=z["X"]; cols=[str(c) for c in z["cols"]]
    idx=[j for j,c in enumerate(cols) if c.startswith(pre)]; sites=[cols[j][len(pre):] for j in idx]
    sids=[str(s) for s in z["sids"]]
    lab={l.split("\t")[0]:l.split("\t")[2] for l in open(MAN).read().splitlines()[1:] if l.split("\t")[1]=="v1"}
    keep=[i for i,s in enumerate(sids) if s in lab]; y=np.array([lab[sids[i]] for i in keep])
    Xk=np.nan_to_num(X[keep].astype(np.float64)[:,idx])
    s2t={}
    for ln in open(REG):
        x=ln.rstrip("\n").split("\t")
        if len(x)>=5: s2t[f"{x[0]}_{(int(x[1])+int(x[2]))//2}"]=x[4]
    tfs=np.array([s2t.get(s,"NA") for s in sites]); TFL=sorted(set(t for t in tfs if t!="NA"))
    T=np.column_stack([Xk[:,tfs==t].mean(1) for t in TFL])
    return T,y,np.array(TFL)


def rows_for(which):
    T,y,TFL=load_pertf(which); tfi={t:j for j,t in enumerate(TFL)}; out=[]
    for c in ROWS:
        yb=(y==c).astype(int)
        clf=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.6,
            reg_lambda=1.0,min_child_weight=2,tree_method="hist",device="cpu",eval_metric="logloss",
            n_jobs=8,verbosity=0).fit(T,yb,sample_weight=compute_sample_weight("balanced",yb))
        contrib=clf.get_booster().predict(xgb.DMatrix(T),pred_contribs=True)[:,:-1]   # TreeSHAP per-sample per-TF
        imp=np.abs(contrib[y==c]).mean(0)                                              # mean |SHAP| on class-c samples
        mtf=[t for t in MASTER[c] if t in tfi]; mi=[tfi[t] for t in mtf]; ri=[j for j in range(len(TFL)) if j not in mi]
        mv=imp[mi]; rv=imp[ri]; p=mannwhitneyu(mv,rv,alternative="greater").pvalue
        topm=", ".join([t for t,_ in sorted(zip(mtf,mv),key=lambda x:-x[1])][:3])
        out.append(dict(channel=which,cancer=c,short=SHORT[c],n_master=len(mtf),
                        master_meanSHAP=float(mv.mean()),bg_meanSHAP=float(rv.mean()),
                        lift=float(mv.mean()-rv.mean()),p=p,stars=stars(p),sig=p<0.05,top_master=topm))
    return pd.DataFrame(out)


def main():
    G=rows_for("gini"); L=rows_for("lenent"); pd.concat([G,L]).to_csv(TSV,sep="\t",index=False)
    for nm,d in (("gini",G),("lenent",L)): print(f"[{nm}] master>rest SHAP sig {int(d.sig.sum())}/{len(d)}",flush=True)
    print(pd.concat([G,L])[["channel","short","master_meanSHAP","bg_meanSHAP","p","stars","top_master"]].round(4).to_string(index=False),flush=True)
    order=G.sort_values("lift",ascending=False).short.tolist()
    col={"***":"#a50f15","**":"#de2d26","*":"#fb6a4a","ns":"grey70"}
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for ax,d,t in zip(axes,(G,L),("endpoint-Gini","fragment-length-entropy")):
        dd=d.set_index("short").reindex(order).reset_index(); yy=np.arange(len(dd))[::-1]
        for i,r in dd.iterrows():
            ax.plot([r.bg_meanSHAP,r.master_meanSHAP],[yy[i],yy[i]],color=col[r.stars],lw=2.4,zorder=1)
            ax.scatter(r.bg_meanSHAP,yy[i],color="#777",s=55,zorder=2)
            ax.scatter(r.master_meanSHAP,yy[i],color=col[r.stars],s=120,zorder=3,edgecolor="k",lw=.5)
            ax.annotate(f"{r.top_master} (p={r.p:.1e})",(r.master_meanSHAP,yy[i]),xytext=(6,0),
                        textcoords="offset points",va="center",fontsize=7,color="#333")
        ax.set_yticks(yy); ax.set_yticklabels([f"{r.short} ({r.stars})" for _,r in dd.iterrows()],fontsize=9)
        ax.set_xlabel("mean |TreeSHAP| contribution to TOO"); ax.set_title(f"{t}\nmaster>rest sig {int(d.sig.sum())}/10",fontweight="bold",fontsize=11); ax.margins(x=0.3); ax.grid(axis="x",alpha=.3)
    axes[0].legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor="#777",markersize=9,label="background TFs"),
        Line2D([0],[0],marker='o',color='w',markerfacecolor="#a50f15",markersize=11,label="master (sig)"),
        Line2D([0],[0],marker='o',color='w',markerfacecolor="grey70",markersize=11,label="master (ns)")],fontsize=8,loc="lower right")
    fig.suptitle("SHAP (XGBoost TreeSHAP) master-TF concordance — per-TF, v1 (nonlinear complement to enet β·z)",fontsize=12,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(OUT,dpi=150); print("WROTE",OUT,flush=True)


if __name__=="__main__": main()
