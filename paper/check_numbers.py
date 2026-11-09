"""Cross-check the paper's numbers against the frozen outputs.

Each key macro of ``numbers.tex`` (written by ``make_numbers.py``) is compared with the verdict line written by the
corresponding pre-registered script in ``results/<study>/summary.txt``, which is produced independently. A macro
differing by more than its tolerance (half a unit of its last printed digit) is reported.

Usage
-----
    python paper/check_numbers.py     # prints "<n> contrôles, 0 écart(s)" when everything matches
"""
import re
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
M = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(Path(__file__).parent / "numbers.tex").read()))
S = {k: open(R / f"{k}/summary.txt").read() for k in ("g2", "v2", "v3", "v4", "v5", "v6", "v7", "v8")}
def num(study, pat):
    """First number captured by regex ``pat`` in the summary of ``study``."""
    return float(re.search(pat, S[study], re.S).group(1))
# (macro, value in the frozen output, tolerance)
checks = [
    ("PhiHatErr", 100 * float(re.search(r"^all .*?phi-hat ([0-9.]+)", open(R / "exploratory_phihat" / "summary.txt").read(), re.M).group(1)), 0.06),
    ("PhiTrueErrSub", 100 * float(re.search(r"^all .*?true phi ([0-9.]+)", open(R / "exploratory_phihat" / "summary.txt").read(), re.M).group(1)), 0.06),
    ("GEight", num("v8", r"N1-v8.*?median x([0-9.]+)"), 0.006),
    ("GEightMin", num("v8", r"N1-v8.*?\(min x([0-9.]+)\)"), 0.006),
    ("GEightMAS", num("v8", r"N2-v8.*?median x([0-9.]+)"), 0.006),
    ("CovEight", num("v8", r"N3-v8.*?mean ([0-9.]+)"), 0.0006),
    ("CovEightMAS", num("v8", r"N3-v8.*?multi-agent ([0-9.]+)"), 0.0006),
    ("LawErrEight", 100 * num("v8", r"N4-v8.*?median error ([0-9.]+)"), 0.06),
    ("BiasNaiveEight", 100 * num("v8", r"A1-v8.*?bias median \+([0-9.]+)"), 0.06),
    ("RmseRatioEight", num("v8", r"A2-v8.*?median ([0-9.]+) ;"), 0.006),
    ("CovDeadlineEight", num("v8", r"A3-v8.*?mean ([0-9.]+)"), 0.0006),
    ("PatPosEight", num("v8", r"P1-v8.*?positive in (\d)/7"), 0.1),
    ("PhiPosEight", num("v8", r"P2-v8.*?positive in (\d)/7"), 0.1),
    ("LawErrPilot", 100 * num("g2", r"H1 m=20 — erreur relative médiane loi\(phi² vrai\) ([0-9.]+)"), 0.06),
    ("LawErrTwo", 100 * num("v2", r"H1-v2.*?loi ([0-9.]+)"), 0.06),
    ("LawErrThree", 100 * num("v3", r"H1-v3.*?loi ([0-9.]+)"), 0.06),
    ("LawErrFour", 100 * num("v4", r"H1-v4.*?phi² ([0-9.]+)"), 0.06),
    ("LawErrFive", 100 * num("v5", r"H1-v5.*?phi² ([0-9.]+)"), 0.06),
    ("LawErrSix", 100 * num("v6", r"U4-v6.*?phi² ([0-9.]+)"), 0.06),
    ("StratErrFour", 100 * num("v4", r"H8-v4.*?médiane ([0-9.]+)"), 0.06),
    ("StratErrFive", 100 * num("v5", r"H8-v5.*?médiane ([0-9.]+)"), 0.06),
    ("StratErrSix", 100 * num("v6", r"U4-v6.*?plan stratifié ([0-9.]+)"), 0.06),
    ("GGraphTwo", num("v2", r"graphe x([0-9.]+)"), 0.006), ("GPredTwo", num("v2", r"seul x([0-9.]+)"), 0.006),
    ("CovCovTwo", num("v2", r"couverture graphe moy ([0-9.]+)"), 0.0006),
    ("GGraphThree", num("v3", r"H5-v3.*?graphe v3 x([0-9.]+)"), 0.006), ("GPredThree", num("v3", r"H5-v3.*?seul x([0-9.]+)"), 0.006),
    ("WorstVThree", num("v3", r"pire gain v3 x([0-9.]+)"), 0.006),
    ("GStratFour", num("v4", r"v3_s x([0-9.]+)"), 0.006), ("GGraphFour", num("v4", r"uniforme \(v3\) x([0-9.]+)"), 0.006),
    ("GPredFour", num("v4", r"H9-v4.*?seul x([0-9.]+)"), 0.006), ("CovCovFour", num("v4", r"H9-v4.*?couverture ([0-9.]+)"), 0.0006),
    ("DecGraphFour", 100 * num("v4", r"erreur de décision v3_s ([0-9.]+)"), 0.06), ("DecResetFour", 100 * num("v4", r"H10-v4.*?reset uniforme ([0-9.]+)"), 0.06),
    ("GGregMinFour", num("v4", r"pire gain x([0-9.]+)"), 0.006), ("CovGregMinFour", num("v4", r"H11-v4.*?min ([0-9.]+)"), 0.0006),
    ("GStratFive", num("v5", r"H9-v5 — v3_s x([0-9.]+)"), 0.006), ("DecGraphFive", 100 * num("v5", r"H10-v5.*?v3_s ([0-9.]+)"), 0.06),
    ("DecResetFive", 100 * num("v5", r"H10-v5.*?reset uniforme ([0-9.]+)"), 0.06),
    ("GGregMinFive", num("v5", r"H12-v5.*?pire gain x([0-9.]+)"), 0.006), ("GGregFive", num("v5", r"H12-v5.*?médiane x([0-9.]+)"), 0.006),
    ("CovGregFive", num("v5", r"H12-v5.*?couverture moy ([0-9.]+)"), 0.0006), ("CovGregMinFive", num("v5", r"H12-v5.*?\(min ([0-9.]+)"), 0.0006),
    ("GGregMinSix", num("v6", r"U1-v6.*?pire gain x([0-9.]+)"), 0.006), ("GGregSix", num("v6", r"U1-v6.*?médiane x([0-9.]+)"), 0.006),
    ("CovGregSix", num("v6", r"U1-v6.*?couverture moy ([0-9.]+)"), 0.0006), ("CovGregMinSix", num("v6", r"U1-v6.*?\(min ([0-9.]+)"), 0.0006),
    ("GPredSix", num("v6", r"U2-v6.*?BMA x([0-9.]+)"), 0.006),
    ("DecGraphSix", 100 * num("v6", r"U3-v6.*?stratifié ([0-9.]+)"), 0.06), ("DecResetSix", 100 * num("v6", r"U3-v6.*?reset uniforme ([0-9.]+)"), 0.06),
    ("DetPower", 100 * num("v3", r"H4-v3.*?alerte ([0-9.]+)"), 0.6), ("DetProbes", num("v3", r"avec ([0-9.]+) probes"), 0.6),
    ("DetFA", 100 * num("v3", r"sans baisse ([0-9.]+)"), 0.06), ("AtlasModel", num("v2", r"A_modele ([0-9.]+)"), 0.006),
    ("AtlasArch", num("v2", r"B_architecture ([0-9.]+)"), 0.006),
    ("SScaffMed", num("v7", r"S1 — .*?écart médian ([+\-0-9.]+)"), 0.0006), ("SScaffPos", num("v7", r"S1 — .*?positif dans ([0-9]+)/7"), 0.1),
    ("SScaffVsModelMed", num("v7", r"S2 — .*?écart médian ([+\-0-9.]+)"), 0.0006), ("SProvGap", num("v7", r"S3 — .*?écart ([+\-0-9.]+)"), 0.0006),
    ("GKnnSeven", num("v7", r"S4 — .*?: x([0-9.]+)"), 0.0006), ("GCfSeven", num("v7", r"S4 — .*?vs x([0-9.]+)"), 0.0006),
    ("KnnRatioSeven", num("v7", r"S4 — .*?rapport médian ([0-9.]+)"), 0.0006), ("AtlasFamily", num("v2", r"C_famille ([0-9.]+)"), 0.006),
]
bad = 0
for name, ref, tol in checks:
    ok = abs(float(M[name]) - ref) <= tol
    bad += not ok
    if not ok:
        print(f"ÉCART {name}: papier {M[name]} / sortie figée {ref}")
print(f"{len(checks)} contrôles, {bad} écart(s)")
