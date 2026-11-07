"""Re-mesurer, avec 10x plus de mondes, les scénarios où le BMA semblait perdre plus de 5 % face au reset."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt.simulate import Scenario, evaluate
src = Path(__file__).resolve().parents[1] / "results" / "g1_heldout_ds_bma_proportional_m20.csv"
rows = [r for r in csv.DictReader(src.open()) if float(r["gain"]) < 0.95]
for r in rows:
    sc = Scenario(r["scenario"], float(r["pi"]), float(r["q1"]), float(r["q0"]))
    b = evaluate(sc, "reset", m=20, reps=4000, seed=7)
    x = evaluate(sc, "ds_bma", m=20, reps=4000, seed=7)
    print(f"{sc.name} pi={sc.pi:.2f} q1={sc.q1:.2f} q0={sc.q0:.2f} phi2={sc.phi2:.3f} | gain 400 mondes x{float(r['gain']):.2f} -> 4000 mondes x{b['mse']/x['mse']:.2f}, couv {x['coverage']:.3f}")
