"""DÉVELOPPEMENT : loi du gain d'un plan stratifié sur la difficulté vue par le graphe (Verified, déjà vu).
Gain exact du plan stratifié (1 probe par strate de taille N/m) sur la moyenne simple :
  G = [(1 - m/N) S² / m] / [(1/m²) Σ_h (1 - 1/N_h) S_h²]  ≈ 1 / (1 - η²_G),  η²_G = part de variance entre strates.
Comparé au gain observé de reset_s dans results/dev_v4_verified.csv."""
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import graph as Gr
import v3_swebench_data as D
M = 20
ids, Y, dates, lin, _ = D.load("/home/claude/data/swe/experiments")
obs = {r["case"]: r for r in csv.DictReader(open(HERE.parent / "results" / "dev_v4_verified.csv"))}
pred, ob, eta = [], [], []
for L, old, new in D.transitions(dates, lin):
    eco = D.ecosystem(dates, lin, old, new)
    if not eco:
        continue
    X = Gr.ecosystem_features(np.array([Y[e] for e in eco]), Y[old]); y = Y[new].astype(float); N = len(y)
    blocks = np.array_split(np.argsort(X[:, 0], kind="stable"), M)
    S2 = y.var(ddof=1)
    if S2 == 0:
        continue
    within = sum((1 - 1 / len(b)) * y[b].var(ddof=1) for b in blocks) / M ** 2
    G = ((1 - M / N) * S2 / M) / within
    e2 = 1 - sum(len(b) * y[b].var() for b in blocks) / (N * y.var())
    pred.append(G); eta.append(e2); ob.append(float(obs[f"{L}|{old}->{new}"]["gain_reset_s"]))
pred, ob, eta = map(np.array, (pred, ob, eta))
err = np.abs(ob / pred - 1)
print(f"{len(ob)} transitions : η²_G médian {np.median(eta):.3f} (q1 {np.quantile(eta,.25):.3f}, q3 {np.quantile(eta,.75):.3f})")
print(f"gain prédit médian x{np.median(pred):.2f} | observé x{np.median(ob):.2f} | erreur relative médiane {np.median(err):.3f}, q90 {np.quantile(err,.9):.3f} | corrélation {np.corrcoef(pred, ob)[0,1]:.3f}")
print(f"approximation 1/(1-η²) : médiane x{np.median(1/(1-eta)):.2f}, erreur médiane vs exact {np.median(np.abs((1/(1-eta))/pred-1)):.3f}")
