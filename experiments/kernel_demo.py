"""Démonstration du noyau sur une vraie lignée (SWE-bench Verified, OpenHands) : l'historique des agents publiés
avant chaque version sert de graphe ; chaque nouvelle version n'est testée que sur les 20 tâches choisies par le
graphe ; le noyau estime, décide et écrit les pages. Sortie : demo/wiki/slots/*.md et demo/decisions.md"""
import shutil, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt.kernel import Kernel
import v3_swebench_data as D

OUT = HERE.parent / "demo"
shutil.rmtree(OUT, ignore_errors=True)
IDS, Y, DATES, LIN, _ = D.load("/home/claude/data/swe/experiments")
iso = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"
K = Kernel(OUT)
K.register_tasks(IDS)
line = sorted([e for e in Y if LIN[e] == "openhands"], key=lambda e: DATES[e])
K.register_slot("openhands", "agent de correction de tickets GitHub (lignée OpenHands, SWE-bench Verified)")
K.register_slot("ecosysteme", "autres agents publiés, mesurés sur toutes les tâches")
rows = []
prev = None
for e in sorted(Y, key=lambda e: (DATES[e], e)):
    if e in line:
        K.register_version("openhands", e, iso(DATES[e]), parent=prev)
        if prev is None:
            K.record_many(e, dict(zip(IDS, Y[e].tolist())))          # première version : mesurée en entier
        else:
            plan = K.plan(e, m=20, seed=1)
            K.record_many(e, {t: bool(Y[e][IDS.index(t)]) for t in plan}, role="probe")
            est, dec = K.estimate(e), K.decide(e)
            rows.append((e, prev, Y[e].mean(), est, dec))
            K.record_many(e, dict(zip(IDS, Y[e].tolist())))          # puis mesurée en entier, pour la suite
        prev = e
    else:
        K.register_version("ecosysteme", e, iso(DATES[e]))
        K.record_many(e, dict(zip(IDS, Y[e].tolist())))
K.render()
L = ["# Décisions du noyau sur la lignée OpenHands (20 tâches testées par version)", "",
     "| Nouvelle version | Version courante | Estimation (intervalle à 90 %) | Vrai taux | Décision | Sûre | Bonne décision |",
     "|---|---|---|---|---|---|---|"]
for e, p, truth, est, dec in rows:
    good = (dec["action"] == "promote") == (truth >= dec["p_current"])
    L.append(f"| {e} | {p} | {est['mean']:.3f} [{est['lo']:.3f} ; {est['hi']:.3f}] | {truth:.3f} | {dec['action']} | "
             f"{'oui' if dec['confident'] else 'non'} | {'oui' if good else 'non'} |")
cov = np.mean([est["lo"] <= t <= est["hi"] for _, _, t, est, _ in rows])
ok = np.mean([(d["action"] == "promote") == (t >= d["p_current"]) for _, _, t, _, d in rows])
L += ["", f"{len(rows)} versions ; intervalle contenant le vrai taux : {cov:.0%} ; bonnes décisions : {ok:.0%} ; "
      f"tâches testées : {20 * len(rows)} au lieu de {500 * len(rows)}."]
(OUT / "decisions.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
