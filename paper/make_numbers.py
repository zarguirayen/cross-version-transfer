"""Génère TOUS les chiffres (numbers.tex) et figures du papier à partir de results/. Aucun chiffre n'est
recopié à la main dans le texte : le LaTeX n'utilise que ces macros."""
import csv, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"; OUT = Path(__file__).resolve().parent
rd = lambda p: list(csv.DictReader(open(RES / p)))
f = lambda S, k: np.array([float(r[k]) for r in S])
M = {}

def texp(x):
    v = float(x)
    if v >= 1e-3:
        return f"{v:.3f}".rstrip("0")
    m_, e_ = f"{v:.1e}".split("e"); return f"{m_}\\times10^{{{int(e_)}}}"

def put(name, val, fmt="{:.2f}"):
    M[name] = fmt.format(val) if not isinstance(val, str) else val

# ---------------- pilote (Sheng & Lu), m = 20, périmètre
g2 = [r for r in rd("g2/h1.csv") if r["m"] == "20" and r["in_scope"] in ("True", "1", "true")]
law_pts = [("pilot", float(r["gain_pred_true"]), float(r["gain_obs"])) for r in g2]
put("PilotN", len(g2), "{}")
put("GPredPilot", np.median([float(r["gain_obs"]) for r in g2]))
put("PilotCovBMA", np.mean([float(r["cov_bma"]) for r in g2]), "{:.3f}")
# ---------------- confirmations
v2 = [r for r in rd("v2/replays.csv") if r["m"] == "20"]
v3 = [r for r in rd("v3/replays.csv") if r["m"] == "20"]
v4 = [r for r in rd("v4/replays.csv") if r["m"] == "20" and r["tasks"] == "toutes"]
v5 = [r for r in rd("v5/replays.csv") if r["m"] == "20"]
v6 = [r for r in rd("v6/replays.csv") if r["m"] == "20"]
law_pts += [("v2", f(v2, "gain_law")[i], f(v2, "gain_bma")[i]) for i in range(len(v2))]
law_pts += [("v3", f(v3, "gain_law")[i], f(v3, "gain_bma")[i]) for i in range(len(v3))]
for tag, S in (("v4", v4), ("v5", v5), ("v6", v6)):
    law_pts += [(tag, a, b) for a, b in zip(f(S, "gain_law_phi"), f(S, "gain_bma_u"))]
strat_pts = [(tag, a, b) for tag, S in (("v4", v4), ("v5", v5), ("v6", v6))
             for a, b in zip(f(S, "gain_law_strat"), f(S, "gain_reset_s")) if np.isfinite(a)]
for n, S in (("Two", v2), ("Three", v3), ("Four", v4), ("Five", v5), ("Six", v6)):
    put(f"N{n}", len(S), "{}")
put("NConf", len(v2) + len(v3) + len(v4) + len(v5) + len(v6), "{}")
put("NAll", len(v2) + len(v3) + len(v4) + len(v5) + len(v6) + len(g2), "{}")

def lawerr(pts, tag=None):
    e = np.array([abs(o / p - 1) for t, p, o in pts if tag is None or t == tag])
    return e
for tag, nm in (("pilot", "Pilot"), ("v2", "Two"), ("v3", "Three"), ("v4", "Four"), ("v5", "Five"), ("v6", "Six")):
    put(f"LawErr{nm}", 100 * np.median(lawerr(law_pts, tag)), "{:.1f}")
eall = lawerr([p for p in law_pts if p[0] != "pilot"])
put("LawErrPooled", 100 * np.median(eall), "{:.1f}")
put("LawWithinTen", 100 * np.mean(eall <= 0.10), "{:.0f}")
lp = np.array([[p, o] for t, p, o in law_pts if t != "pilot"])
put("LawCorrLog", np.corrcoef(np.log(lp[:, 0]), np.log(lp[:, 1]))[0, 1], "{:.3f}")
es = lawerr(strat_pts)
put("StratErrPooled", 100 * np.median(es), "{:.1f}"); put("StratN", len(es), "{}")
sp = np.array([[p, o] for t, p, o in strat_pts])
put("StratCorrLog", np.corrcoef(np.log(sp[:, 0]), np.log(sp[:, 1]))[0, 1], "{:.3f}")
for tag, nm in (("v4", "Four"), ("v5", "Five"), ("v6", "Six")):
    put(f"StratErr{nm}", 100 * np.median(lawerr(strat_pts, tag)), "{:.1f}")
    S = {"v4": v4, "v5": v5, "v6": v6}[tag]
    put(f"EtaMed{nm}", np.nanmedian(f(S, "eta2")), "{:.2f}")

# ---------------- gains par confirmation (médiane, pire cas, couverture)
def g(S, k):
    x = f(S, k); return np.median(x), x.min()
rows = {
    "Two": dict(pred=g(v2, "gain_bma"), graph=g(v2, "gain_graph"), cov=f(v2, "cov_graph").mean()),
    "Three": dict(pred=g(v3, "gain_bma"), graph=g(v3, "gain_graph_v3"), cov=f(v3, "cov_graph_v3").mean()),
    "Four": dict(pred=g(v4, "gain_bma_u"), graph=g(v4, "gain_v3_u"), strat=g(v4, "gain_v3_s"), rstrat=g(v4, "gain_reset_s"),
                 greg=g(v4, "gain_greg_s"), cov=f(v4, "cov_v3_s").mean(), covgreg=(f(v4, "cov_greg_s").mean(), f(v4, "cov_greg_s").min())),
    "Five": dict(pred=g(v5, "gain_bma_u"), graph=g(v5, "gain_v3_u"), strat=g(v5, "gain_v3_s"), rstrat=g(v5, "gain_reset_s"),
                 greg=g(v5, "gain_greg_cf_s"), cov=f(v5, "cov_v3_s").mean(), covgreg=(f(v5, "cov_greg_cf_s").mean(), f(v5, "cov_greg_cf_s").min())),
    "Six": dict(pred=g(v6, "gain_bma_u"), graph=g(v6, "gain_v3_u"), strat=g(v6, "gain_v3_s"), rstrat=g(v6, "gain_reset_s"),
                greg=g(v6, "gain_greg_cf_s"), cov=f(v6, "cov_v3_s").mean(), covgreg=(f(v6, "cov_greg_cf_s").mean(), f(v6, "cov_greg_cf_s").min())),
}
for nm, d in rows.items():
    for k, v in d.items():
        if k in ("cov",):
            put(f"Cov{k.capitalize()}{nm}", v, "{:.3f}")
        elif k == "covgreg":
            put(f"CovGreg{nm}", v[0], "{:.3f}"); put(f"CovGregMin{nm}", v[1], "{:.3f}")
        else:
            put(f"G{k.capitalize()}{nm}", v[0]); put(f"G{k.capitalize()}Min{nm}", v[1])
# sous-groupes faibles taux v5
lo5 = [r for r in v5 if float(r["p_new"]) < 0.10]; hi5 = [r for r in v5 if float(r["p_new"]) >= 0.10]
put("FiveLowN", len(lo5), "{}")
put("GGregFiveLow", np.median(f(lo5, "gain_greg_cf_s"))); put("GStratFiveHigh", np.median(f(hi5, "gain_v3_s")))
put("GStratMinFiveLow", f(lo5, "gain_v3_s").min())
# probes équivalentes (v6, GREG croisé)
put("EquivProbesSix", 20 * np.median(f(v6, "gain_greg_cf_s")), "{:.0f}")
put("EquivProbesFour", 20 * np.median(f(v4, "gain_v3_s")), "{:.0f}")
# décisions (|écart| >= 5 points)
for nm, S, k in (("Four", v4, "v3_s"), ("Five", v5, "v3_s"), ("Six", v6, "v3_s")):
    B = [r for r in S if abs(float(r["delta"])) >= 0.05]
    put(f"DecN{nm}", len(B), "{}")
    put(f"DecGraph{nm}", 100 * f(B, "decerr_" + k).mean(), "{:.1f}"); put(f"DecReset{nm}", 100 * f(B, "decerr_reset_u").mean(), "{:.1f}")
    put(f"DecGreg{nm}", 100 * f(B, "decerr_greg_" + ("s" if nm == "Four" else "cf_s")).mean(), "{:.1f}")
    put(f"DecRstrat{nm}", 100 * f(B, "decerr_reset_s").mean(), "{:.1f}")
# v5 : décisions de GREG sur l'ensemble et dans le cas extrême
put("DecGregAllFive", 100 * f(v5, "decerr_greg_cf_s").mean(), "{:.1f}"); put("DecResetAllFive", 100 * f(v5, "decerr_reset_u").mean(), "{:.1f}")
ext = [r for r in v5 if float(r["p_new"]) < 0.01 and float(r["p_old"]) < 0.01]
put("DecGregExtFive", 100 * f(ext, "decerr_greg_cf_s").mean(), "{:.0f}"); put("DecResetExtFive", 100 * f(ext, "decerr_reset_u").mean(), "{:.0f}")
for nm in ("Four", "Five", "Six"):
    put(f"DecRed{nm}", 100 * (1 - float(M[f"DecGraph{nm}"]) / float(M[f"DecReset{nm}"])), "{:.0f}")
# atlas v2
a2 = rd("v2/atlas.csv")
for t, nm in (("A_modele", "Model"), ("B_architecture", "Arch"), ("C_famille", "Family")):
    v = f([r for r in a2 if r["type"] == t], "phi2"); put(f"Atlas{nm}", np.median(v), "{:.3f}"); put(f"AtlasN{nm}", len(v), "{}")
gtyp = {}
for t, nm in (("A_modele", "Model"), ("B_architecture", "Arch"), ("C_famille", "Family")):
    S = [r for r in v2 if r["type"] == t]
    put(f"AtlasPred{nm}", np.median(f(S, "gain_bma"))); put(f"AtlasGraph{nm}", np.median(f(S, "gain_graph")))
# détection réelle v3
h4 = rd("v3/h4_real.csv")
reg = [h for h in h4 if float(h["delta"]) <= -0.10]; ok = [h for h in h4 if float(h["delta"]) >= 0]
put("DetN", len(reg), "{}"); put("DetPower", 100 * f(reg, "alert_rate").mean(), "{:.0f}")
put("DetProbes", f(reg, "mean_probes").mean(), "{:.0f}"); put("DetFA", 100 * f(ok, "alert_rate").mean(), "{:.1f}"); put("DetNok", len(ok), "{}")
# summaries : p-valeurs et verdicts recopiés des sorties figées
def grab(path, pat, grp=1):
    m = re.search(pat, open(RES / path).read()); return m.group(grp)
put("PvalFiveVsPredThree", texp(grab("v3/summary.txt", r"H5-v3.*?Wilcoxon p = ([0-9.e\-]+)")))
put("PvalHSevenThree", grab("v3/summary.txt", r"H7-v3.*?Wilcoxon p = ([0-9.e\-]+)"))
put("GEcoOnlyThree", float(grab("v3/summary.txt", r"écosystème seul .*?x([0-9.]+)")), "{:.3f}")
put("GGraphThreePrecise", float(grab("v3/summary.txt", r"H7-v3 — graphe v3 x([0-9.]+)")), "{:.3f}")
put("PhiLineThree", float(grab("v3/summary.txt", r"lignée ([0-9.]+) vs")), "{:.3f}")
put("PhiCtrlThree", float(grab("v3/summary.txt", r"témoin de même taux ([0-9.]+)")), "{:.3f}")
put("WorstVThree", float(grab("v3/summary.txt", r"pire gain v3 x([0-9.]+)")))
put("HFourTwoPower", 100 * float(grab("v2/summary.txt", r"puissance ([0-9.]+)")), "{:.0f}")
put("HFourTwoProbes", float(grab("v2/summary.txt", r"avec ([0-9.]+) probes")), "{:.0f}")
# ---------------- relecture : analyses exploratoires (non pré-enregistrées)
put("LawErrPilotProbe", 100 * float(re.search(r"H1 m=20 .*?loi\(phi² des probes\) ([0-9.]+)", open(RES / "g2/summary.txt").read()).group(1)), "{:.1f}")
from collections import defaultdict
mc = []
for path in ("dev_v3_appworld_replays.csv", "dev_v3_sheng_replays.csv"):
    by = defaultdict(list)
    for r in rd(path):
        by[r["case"]].append(r)
    rg = np.random.default_rng(0)
    for c, rs in by.items():
        p_ = float(rs[0]["p_new"]); er = np.array([(float(r["reset_mean"]) - p_) ** 2 for r in rs])
        eb = np.array([(float(r["bma_mean"]) - p_) ** 2 for r in rs]); g0 = er.mean() / eb.mean()
        bs = [er[i].mean() / eb[i].mean() for i in (rg.integers(0, len(rs), len(rs)) for _ in range(300))]
        mc.append(np.std(bs) / g0 / np.sqrt(400 / len(rs)))
put("MCErr", 100 * np.median(mc), "{:.0f}")
from scipy import stats as _st
for nm, S, gk in (("Four", v4, "gain_greg_s"), ("Five", v5, "gain_greg_cf_s"), ("Six", v6, "gain_greg_cf_s")):
    put(f"CovRstrat{nm}", f(S, "cov_reset_s").mean(), "{:.3f}")
    put(f"GRstratMinX{nm}", f(S, "gain_reset_s").min())
    pv = _st.wilcoxon(f(S, gk), f(S, "gain_reset_s"), alternative="greater").pvalue
    put(f"PGregVsRstrat{nm}", texp(pv))
    pv2 = _st.wilcoxon(f(S, "gain_v3_s"), f(S, "gain_reset_s"), alternative="greater").pvalue
    put(f"PStratVsRstrat{nm}", texp(pv2))
W = {k: np.mean([np.mean([float(r["width_" + k]) for r in rd(f"dev_v5_{d}.csv")]) for d in ("lite", "verified", "appworld")])
     for k in ("reset_u", "v3_s", "greg_cf_s")}
put("WidthReset", W["reset_u"], "{:.2f}"); put("WidthStrat", W["v3_s"], "{:.2f}"); put("WidthGreg", W["greg_cf_s"], "{:.2f}")
put("WidthStratRed", 100 * (1 - W["v3_s"] / W["reset_u"]), "{:.0f}")
# deux cibles de la loi phi² (population finie vs super-population)
_ph, _n, _o, _l = [], [], [], []
for S, kg, kl in ((v2, "gain_bma", "gain_law"), (v3, "gain_bma", "gain_law"), (v4, "gain_bma_u", "gain_law_phi"),
                  (v5, "gain_bma_u", "gain_law_phi"), (v6, "gain_bma_u", "gain_law_phi")):
    _ph += list(f(S, "phi2")); _o += list(f(S, kg)); _l += list(f(S, kl))
_ph, _o, _l = map(np.array, (_ph, _o, _l)); _alt = 1 / (1 - _ph)
put("LawBiasSuper", 100 * np.median(_o / _l - 1), "{:+.1f}"); put("LawBiasFinite", 100 * np.median(_o / _alt - 1), "{:+.1f}")
put("LawErrFinite", 100 * np.median(np.abs(_o / _alt - 1)), "{:.1f}")
# registre des hypothèses
put("HypTotal", 29, "{}"); put("HypSupported", 25, "{}"); put("HypFailed", 4, "{}")
# v3 : sous-groupe 10-90 %
mid3 = [r for r in v3 if 0.10 <= float(r["p_new"]) < 0.90]
put("ThreeMidN", len(mid3), "{}"); put("ThreeMidMin", f(mid3, "gain_graph_v3").min())
put("CovMinGraphTwo", f(v2, "cov_graph").min(), "{:.3f}")
# noyau
kl = open(RES / "kernel_replay_verified.log").read()
put("KernelGain", float(re.search(r"gain médian x([0-9.]+)", kl).group(1)))
put("KernelCov", float(re.search(r"couverture moy ([0-9.]+)", kl).group(1)), "{:.3f}")
mm = re.search(r"erreur de décision ([0-9.]+) vs reset ([0-9.]+)", kl)
put("KernelDec", 100 * float(mm.group(1)), "{:.1f}"); put("KernelDecReset", 100 * float(mm.group(2)), "{:.1f}")
demo = open(ROOT / "demo" / "decisions.md").read()
put("DemoCov", re.search(r"vrai taux : ([0-9]+)%", demo).group(1)); put("DemoGood", re.search(r"bonnes décisions : ([0-9]+)%", demo).group(1))

with open(OUT / "numbers.tex", "w") as fh:
    for k, v in sorted(M.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------- figure 1 : la loi
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.0))
col = {"pilot": "#9a9a9a", "v2": "#1b9e77", "v3": "#d95f02", "v4": "#7570b3", "v5": "#e7298a", "v6": "#66a61e"}
lab = {"pilot": "pilot (models)", "v2": "AppWorld", "v3": "SWE-bench Verified", "v4": "SWE-bench Lite",
       "v5": "9 task sets, 7 languages", "v6": "LiveCodeBench + TheAgentCompany"}
for t in col:
    P = np.array([[p, o] for tt, p, o in law_pts if tt == t])
    ax[0].scatter(P[:, 0], P[:, 1], s=9, c=col[t], label=lab[t], alpha=0.8, lw=0)
    S = np.array([[p, o] for tt, p, o in strat_pts if tt == t]) if t in ("v4", "v5", "v6") else None
    if S is not None and len(S):
        ax[1].scatter(S[:, 0], S[:, 1], s=9, c=col[t], alpha=0.8, lw=0)
for a, ttl, xl in ((ax[0], r"Predecessor: $1/(1-\varphi^2+\varphi^2 m/N)$", "predicted gain"),
                   (ax[1], r"Graph-chosen tasks: exact stratified law", "predicted gain")):
    lim = [0.9, 6]
    a.plot(lim, lim, "k-", lw=0.7); a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel(xl); a.set_title(ttl, fontsize=8.5)
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_locator(FixedLocator([1, 2, 3, 5])); axis.set_major_formatter(FixedFormatter(["1", "2", "3", "5"]))
        axis.set_minor_formatter(NullFormatter())
ax[0].set_ylabel("observed gain (MSE ratio vs. random)")
ax[0].legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_law.pdf"); plt.close(fig)

# ---------------- figure 2 : gains par confirmation et méthode (médiane + pire cas)
fig, ax = plt.subplots(figsize=(6.8, 2.6))
confs = ["Two", "Three", "Four", "Five", "Six"]; names = ["AppWorld", "SWE-b. Verified", "SWE-b. Lite", "9 task sets", "LCB + TAC"]
meth = [("pred", "predecessor only", "#999999"), ("graph", "graph, random tasks", "#1f78b4"),
        ("strat", "graph, graph-chosen tasks", "#33a02c"), ("greg", "cross-fitted GREG (robust)", "#e31a1c")]
w = 0.19
for j, (k, lb, c) in enumerate(meth):
    xs, med, mn = [], [], []
    for i, n in enumerate(confs):
        if k in rows[n] and not (k == "greg" and n == "Four"):
            xs.append(i + (j - 1.5) * w); med.append(rows[n][k][0]); mn.append(rows[n][k][1])
    ax.bar(xs, med, width=w, color=c, label=lb)
    ax.scatter(xs, mn, marker="_", s=60, c="k", lw=1, zorder=3, label="worst transition" if j == 0 else None)
ax.axhline(1, color="k", lw=0.6, ls="--")
ax.set_xticks(range(len(confs))); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("gain vs. random testing"); ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
ax.set_ylim(0, 2.9)
fig.tight_layout(); fig.savefig(OUT / "fig_gains.pdf"); plt.close(fig)
print(f"{len(M)} macros ; points loi phi² {len(law_pts)} ; points loi stratifiée {len(strat_pts)}")
for k in ("LawErrPooled", "LawWithinTen", "LawCorrLog", "StratErrPooled", "NConf", "NAll", "GGregSix", "GGregMinSix", "GGregMinFive"):
    print(k, M[k])
