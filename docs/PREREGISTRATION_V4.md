# Pré-enregistrement v4 — le graphe choisit les tâches (SWE-bench Lite)

**Statut : figé par le tag git `prereg-v4` avant toute statistique de résultat sur ces données.**
Rédigé le 7 novembre 2026. Analyse figée dans `experiments/v4_lite_preregistered.py` et
`experiments/v4_lite_data.py` (même commit). Testée à blanc avec `--fake`.

## Pourquoi une v4

Jusqu'ici le graphe servait à **estimer** à partir de tâches tirées au hasard. Question nouvelle : le graphe
doit-il aussi **choisir** les tâches ? Plan proposé : trier les N tâches par difficulté vue par l'écosystème
(agents antérieurs uniquement), former m strates contiguës de taille ~N/m, tester 1 tâche tirée au hasard par
strate. La sélection ne dépend que de l'historique ; la moyenne simple reste sans biais, et l'inférence fondée
sur le modèle reste valide (sélection ignorable car fonction des seules covariables).

**Extension de la loi.** Le gain exact du plan stratifié sur la moyenne simple est
G = [(1 − m/N)·S²/m] / [(1/m²)·Σ_h (1 − 1/N_h)·S_h²] ≈ 1/(1 − η²_G), où η²_G est la part de variance des
résultats de v′ expliquée par les strates de difficulté du graphe. C'est la même loi que pour le
prédécesseur (1/(1 − φ²)), avec une autre mesure de l'information portée par le graphe.

**Développement, sur données déjà vues uniquement** (R = 200, m = 20 ; `results/dev_v4*`) :

| Données | reset uniforme | graphe v3, uniforme | moyenne simple, stratifié | graphe v3, stratifié | GREG, stratifié |
|---|---|---|---|---|---|
| SWE-bench Verified, 89 | ×1,00 | ×2,05–2,08 | ×2,19–2,26 | **×2,41–2,46** (min ×0,69–0,71) | ×2,15 (min ×1,19 ; couverture 0,914, min 0,78) |
| AppWorld, 40 | ×1,00 | ×1,30 | ×1,31–1,33 | **×1,48** (min ×0,54–0,59) | ×1,39 (min ×0,85 ; couverture 0,903, min 0,765) |

Loi du plan stratifié sur Verified : gain prédit ×2,29, observé ×2,27, erreur médiane 9 %, η²_G médian 0,58.
Le graphe stratifié (`v3_s`) a le meilleur gain et des intervalles calibrés ; l'estimateur assisté par modèle
(`greg_s`, `graph_greg` : prédictions du graphe corrigées par les résidus observés, sans biais quel que soit le
modèle) ne perd jamais beaucoup, y compris sous 10 % de réussite, mais ses intervalles sont trop étroits.
Choix fixé avant le gel : `v3_s` principal ; `greg_s` testé uniquement sur la robustesse (H11).

## Données

- `SWE-bench/experiments`, `evaluation/lite`, même clone que la v3 : 84 soumissions, 300 tâches, aucun fichier de
  résultats Lite ouvert avant ce gel (audit de forme : clés des fichiers, univers de 300 tâches, 93 communes avec
  Verified, noms et dates).
- 28 transitions de lignée (règle mécanique de `v4_lite_data.py`, liste dans le script). Écosystème = soumissions
  Lite antérieures à v′, sauf v.

## Protocole

R = 400 rejeux par transition et par budget m ∈ {20 (principal), 40} ; sur les 300 tâches (principal) et sur les
207 tâches absentes de Verified (secondaire). Plans : uniforme (`_u`) et stratifié (`_s`), tirés indépendamment
à chaque rejeu. Estimateurs : `reset_u`, `bma_u`, `v3_u` (méthode v3 confirmée), `reset_s`, `v3_s`, `greg_s`.
Graine 20261116.

## Hypothèses confirmatoires et critères (m = 20, 300 tâches, 28 transitions)

| | Mesure | Critère |
|---|---|---|
| **H1-v4 — La loi φ² tient (4e réplication)** | \|gain du BMA / loi finie − 1\| | médiane ≤ 0,20 |
| **H8-v4 — La loi s'étend au plan stratifié** | \|gain de `reset_s` / loi exacte du plan − 1\| | médiane ≤ 0,20 |
| **H9-v4 — Choisir les tâches avec le graphe fait mieux** | gain de `v3_s` contre gain de `v3_u` | Wilcoxon apparié unilatéral p < 0,05 **et** `v3_s` ≥ 0,95 × reset sur ≥ 90 % **et** couverture moyenne ≥ 0,88 |
| **H10-v4 — Meilleures décisions** | erreur de décision « garder v ou passer à v′ » (signe de p̂′ − p_v), transitions où \|p′ − p_v\| ≥ 0,05 | Wilcoxon apparié unilatéral, `v3_s` < `reset_u`, p < 0,05 |
| **H11-v4 — Pas de perte catastrophique (estimateur robuste)** | pire gain de `greg_s` | ≥ 0,80 |

Rapportés sans critère : m = 40 ; 207 tâches hors Verified ; couvertures de tous les estimateurs ; η²_G ; pire
cas de `v3_s` ; sous-groupe des versions sous 10 %.

## Limites connues avant les résultats

- 28 transitions seulement ; les deux transitions RAG ont un écosystème de 3 agents.
- Les agents de Lite recouvrent en partie ceux de Verified (mêmes produits) ; les méthodes ont été développées sur
  Verified ; 93 tâches sont communes (d'où l'analyse secondaire sur les 207 autres).
- Plan stratifié à 1 probe par strate : pas d'estimateur de variance sans biais pour la moyenne simple ; les
  intervalles de `reset_s` (Clopper-Pearson) sont conservateurs.
