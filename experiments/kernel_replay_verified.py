"""VALIDATION DU NOYAU (étape 2) : rejeu de bout en bout sur SWE-bench Verified (données déjà vues, servent ici à
valider le logiciel, pas la méthode). Pour chaque transition v -> v', on construit un noyau contenant les versions
antérieures à v' (historique complet), v et v' ; le noyau choisit les 20 tâches (plan), on enregistre seulement
leurs résultats, puis estimate et decide. Comparaison avec la vérité et avec le reset uniforme.
Critères fixés avant exécution : gain médian à ±10 % de ×2,11 (valeur de recherche, dev_v5) ; couverture moyenne
>= 0,88 ; erreur de décision (écart >= 5 points) inférieure à celle du reset uniforme."""
import shutil, sys, tempfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import estimators as E
from cvt.kernel import Kernel
import v3_swebench_data as D

R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
IDS, Y, DATES, LIN, _ = D.load("/home/claude/data/swe/experiments")
TRANS = D.transitions(DATES, LIN)
iso = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"


def run(arg):
    k, (lin, old, new) = arg
    tmp = Path(tempfile.mkdtemp())
    K = Kernel(tmp)
    K.register_tasks(IDS)
    keep = [e for e in Y if DATES[e] < DATES[new]] + [new]
    for e in keep:
        K.register_slot(LIN[e] or e)
    for e in sorted(keep, key=lambda e: DATES[e]):
        K.register_version(LIN[e] or e, e, iso(DATES[e]), parent=old if e == new else None)
        if e != new:
            K.record_many(e, dict(zip(IDS, Y[e].tolist())))
    y2 = Y[new]; p = y2.mean(); pv = Y[old].mean(); truth = dict(zip(IDS, y2.tolist()))
    rng = np.random.default_rng([7, k])
    err_k, err_r, cov, dk, dr = [], [], [], [], []
    for rep in range(R):
        plan = K.plan(new, m=20, seed=int(rng.integers(1 << 30)))
        K.results[new] = {}                                   # nouvelle campagne de probes (état en mémoire)
        K.record_many(new, {t: truth[t] for t in plan}, role="probe")
        e = K.estimate(new); d = K.decide(new)
        iu = rng.choice(len(IDS), 20, replace=False); s = int(y2[iu].sum())
        r = E.reset(s, 0, 0, 20 - s, rng=rng, draws=1000, prior="hybrid")
        err_k.append((e["mean"] - p) ** 2); err_r.append((r.mean - p) ** 2); cov.append(e["lo"] <= p <= e["hi"])
        dk.append((d["action"] == "promote") != (p >= pv)); dr.append((r.mean >= pv) != (p >= pv))
    shutil.rmtree(tmp)
    lint = K.lint()
    print(f"fait : {new[:50]}", flush=True)
    return dict(new=new, p=p, delta=p - pv, gain=np.mean(err_r) / np.mean(err_k), cov=np.mean(cov),
                dec_k=np.mean(dk), dec_r=np.mean(dr), lint=len([f for f in lint if f["code"] != "L3"]))


if __name__ == "__main__":
    with Pool(2) as pool:
        rows = pool.map(run, list(enumerate(TRANS)), chunksize=1)
    g = np.array([r["gain"] for r in rows]); c = np.array([r["cov"] for r in rows])
    big = [r for r in rows if abs(r["delta"]) >= 0.05]
    dk = np.mean([r["dec_k"] for r in big]); dr = np.mean([r["dec_r"] for r in big])
    print(f"\n{len(rows)} transitions, R={R} : gain médian x{np.median(g):.3f} (min x{g.min():.2f}) ; couverture moy {c.mean():.3f} "
          f"(min {c.min():.3f}) ; erreur de décision {dk:.3f} vs reset {dr:.3f} ({len(big)} transitions) ; anomalies lint {sum(r['lint'] for r in rows)}")
    ok = abs(np.median(g) / 2.11 - 1) <= 0.10 and c.mean() >= 0.88 and dk < dr
    print(f"VALIDATION DU NOYAU : {'réussie' if ok else 'échouée'}")
