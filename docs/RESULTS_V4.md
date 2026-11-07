# Résultats v4 — le graphe choisit les tâches (SWE-bench Lite)

Exécutée le 7 novembre 2026 après le gel `prereg-v4` (empreintes vérifiées après exécution :
`docs/PREREG_V4_SHA256.txt`). Script figé : `experiments/v4_lite_preregistered.py`, R = 400, graine 20261116.
Données jamais analysées : 84 agents de SWE-bench Lite, 300 tâches, 28 transitions de lignée en ordre temporel
réel. Sorties : `results/v4/`.

## Verdict, selon les critères figés (m = 20, 300 tâches)

| | Critère | Résultat | Verdict |
|---|---|---|---|
| **H1-v4 — La loi φ² tient (4e réplication)** | erreur médiane ≤ 0,20 | 0,039 | soutenue |
| **H8-v4 — La loi s'étend au plan stratifié** | erreur médiane ≤ 0,20 | **0,088** (η²_G médian 0,54) | soutenue |
| **H9-v4 — Choisir les tâches avec le graphe fait mieux** | Wilcoxon p < 0,05 ; ≥ 0,95 × reset sur ≥ 90 % ; couverture ≥ 0,88 | **×2,14** contre ×1,95 (graphe, tâches au hasard) et ×1,47 (prédécesseur seul) ; p = 1,6 × 10⁻⁷ ; 96 % ; couverture 0,948 | soutenue |
| **H10-v4 — Meilleures décisions de version** | Wilcoxon p < 0,05 | erreur « garder v ou passer à v′ » **0,065 contre 0,141** sur 21 transitions | soutenue |
| **H11-v4 — Pas de perte catastrophique (GREG)** | pire gain ≥ 0,80 | **×1,06** (médiane ×1,84) | soutenue |

## Ce que montrent les données

**Une seule loi.** Le gain d'efficacité vaut 1/(1 − ρ²), où ρ² mesure l'information que le graphe porte sur la
nouvelle version : φ² avec le prédécesseur (double échantillonnage), η²_G avec la difficulté des tâches vue par
l'écosystème (plan stratifié). Les deux formes sont confirmées sur des données neuves, à 4 % et 9 % près.

**Le graphe sert deux fois : pour choisir les tâches et pour estimer.** Rien qu'en répartissant les 20 tâches sur
l'échelle de difficulté du graphe, la simple moyenne passe de ×1,00 à ×1,93. Avec l'estimateur graphe en plus :
×2,14. Les 20 tâches valent alors 43 tâches tirées au hasard.

**Les décisions de version sont deux fois plus sûres.** Avec 20 tâches, on se trompe sur « la nouvelle version
est-elle meilleure que l'ancienne ? » dans 14,1 % des cas en tirant au hasard, 6,5 % avec le graphe (transitions
où l'écart réel dépasse 5 points).

**Le problème des versions faibles est résolu pour l'estimation ponctuelle, pas pour les intervalles.** Avec le
plan stratifié, l'estimateur assisté par modèle (GREG) ne perd jamais face au reset : pire cas ×1,06, y compris
sur RAG + GPT-4 (2,7 % de réussite, écosystème de 3 agents : ×1,52). Mais ses intervalles sont trop étroits
(couverture moyenne 0,907, minimum 0,625). L'estimateur principal `v3_s` a des intervalles calibrés et un pire cas
×0,91, bien meilleur que la méthode v3 (×0,76).

**Robustesse.** Sur les 207 tâches absentes de Verified, les résultats sont les mêmes (×2,18 ; couverture 0,954).
À 40 tâches : ×2,14 ; erreur de décision 5,4 % contre 12,5 %.

## Bilan des quatre confirmations

| Étape | Données | Transitions | Prédécesseur seul | Graphe | Meilleure méthode |
|---|---|---|---|---|---|
| v1 | modèles, Sheng & Lu | 30 | ×1,26 | — | — |
| v2 | agents AppWorld | 40 | ×1,12 | ×1,40 | graphe, tâches au hasard |
| v3 | agents SWE-bench Verified, ordre réel | 89 | ×1,53 | ×2,01 | graphe, tâches au hasard |
| v4 | agents SWE-bench Lite, ordre réel | 28 | ×1,47 | ×1,95 | **graphe qui choisit les tâches : ×2,14** |

## Limites

- 28 transitions ; deux transitions RAG avec un écosystème de 3 agents.
- Agents de Lite et de Verified en partie issus des mêmes produits ; méthode développée sur Verified (l'analyse sur
  les 207 tâches non communes donne les mêmes résultats).
- Les intervalles de GREG ne sont pas calibrés ; ceux de la moyenne stratifiée sont trop larges (couverture 0,985).
- Travaux proches à citer : sélection de tâches par IRT (tinyBenchmarks, anchor points), Agent Psychometrics,
  Efficient Benchmarking in Production, Task-Efficient Agent Evaluation Using Cached Traces.
